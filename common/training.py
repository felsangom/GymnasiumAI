"""
Loop de treino, avaliação greedy e linha de comando compartilhados.

Conceitos:
- "Reward" é a recompensa usada no treino (pode conter reward shaping).
- "Score" é a recompensa original do ambiente (`info["raw_reward"]` quando um
  wrapper altera a recompensa). Checkpoint e critério de parada usam o Score
  de avaliações greedy periódicas (epsilon=0) em um ambiente separado.
- Os pesos em disco só são substituídos quando a avaliação supera a pontuação
  registrada no arquivo `.meta.json` ao lado deles.
"""
import argparse
import datetime
import json
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf

from common.dqn import get_render_mode, set_seed, try_load_weights


@dataclass
class TrainConfig:
    name: str
    weights_file: Path
    log_dir: Path
    num_episodes: int
    target_score: Optional[float] = None   # Score médio greedy para considerar resolvido
    eval_interval: int = 20                # avalia a cada N episódios de treino
    eval_episodes: int = 10
    reward_window_size: int = 100          # média móvel exibida no log
    seed: int = 42
    epsilon_when_loaded: float = 0.05      # epsilon inicial ao treinar com --load

    @property
    def meta_file(self):
        return self.weights_file.with_name(self.weights_file.name.replace(".weights.h5", ".meta.json"))

    @property
    def state_file(self):
        return self.weights_file.with_name(self.weights_file.name.replace(".weights.h5", ".train_state.npz"))

    @property
    def eval_seed(self):
        return self.seed + 1000


class _Logger:
    """TensorBoard opcional: se não estiver instalado, o treino segue sem logs."""
    def __init__(self, log_dir):
        self.writer = None
        try:
            stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            self.writer = tf.summary.create_file_writer(str(Path(log_dir) / stamp))
            with self.writer.as_default():
                tf.summary.scalar('Epsilon', 1.0, step=0)
        except Exception as e:
            self.writer = None
            print(f"⚠️ TensorBoard indisponível ({type(e).__name__}); treinando sem logs. "
                  f"Instale com: pip install tensorboard")

    def scalars(self, step, **values):
        if self.writer is None:
            return
        with self.writer.as_default():
            for key, value in values.items():
                if value is not None:
                    tf.summary.scalar(key, value, step=step)

    def close(self):
        if self.writer is not None:
            self.writer.flush()


# =============================================================================
# AVALIAÇÃO
# =============================================================================
def evaluate(env, agent, episodes=10, seed=None, verbose=True):
    """
    Roda a política puramente gulosa (epsilon=0) e devolve (score médio, scores).
    Com `seed`, os episódios usam sementes fixas (seed, seed+1, ...), o que torna
    avaliações diferentes comparáveis entre si.
    """
    scores = []
    for ep in range(episodes):
        state, _ = env.reset(seed=seed + ep if seed is not None else None)
        done, score = False, 0.0
        while not done:
            state, reward, terminated, truncated, info = env.step(agent.greedy_action(state))
            done = terminated or truncated
            score += info.get("raw_reward", reward)
        scores.append(score)
    mean = float(np.mean(scores))
    if verbose:
        print(f"📊 Avaliação greedy ({episodes} ep.): score médio={mean:.1f} | "
              f"min={min(scores):.1f} | max={max(scores):.1f}")
    return mean, scores


def _read_meta(cfg):
    try:
        return json.loads(cfg.meta_file.read_text())
    except (OSError, ValueError):
        return None


def _write_meta(cfg, eval_mean, trained_episodes=None):
    cfg.meta_file.write_text(json.dumps({
        "eval_mean": eval_mean,
        "eval_episodes": cfg.eval_episodes,
        "eval_seed": cfg.eval_seed,
        "trained_episodes": trained_episodes,
        "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }, indent=2) + "\n")


def _existing_best_score(cfg, eval_env, build_agent):
    """Pontuação dos pesos já em disco (para não sobrescrevê-los com algo pior)."""
    if not cfg.weights_file.exists():
        return -np.inf
    meta = _read_meta(cfg)
    same_protocol = meta and (
        meta.get("eval_episodes", cfg.eval_episodes) == cfg.eval_episodes
        and meta.get("eval_seed", cfg.eval_seed) == cfg.eval_seed
    )
    if meta and "eval_mean" in meta and same_protocol:
        score = float(meta["eval_mean"])
        print(f"🔒 Pesos existentes com score {score:.1f}; só serão substituídos por um modelo melhor.")
        return score
    print("🔎 Pesos existentes sem pontuação registrada no protocolo de avaliação atual; "
          "avaliando para usar como referência...")
    probe = build_agent(eval_env)
    if not try_load_weights([probe.q_network], cfg.weights_file):
        print("   Pesos incompatíveis com a arquitetura atual; serão substituídos no primeiro checkpoint.")
        return -np.inf
    score, _ = evaluate(eval_env, probe, cfg.eval_episodes, seed=cfg.eval_seed)
    _write_meta(cfg, score)
    print(f"🔒 Pesos existentes com score {score:.1f}; só serão substituídos por um modelo melhor.")
    return score


# =============================================================================
# TREINO
# =============================================================================
def train(env, eval_env, agent, cfg, *, file_best=-np.inf, start_episode=0, run_best=-np.inf):
    """
    Treina o agente. A cada `eval_interval` episódios faz uma avaliação greedy;
    se o score superar `file_best`, salva os pesos. O estado completo do treino
    é salvo a cada avaliação e ao sair (inclusive com Ctrl+C), para `--resume`.
    """
    logger = _Logger(cfg.log_dir)
    reward_window = deque(maxlen=cfg.reward_window_size)
    episode = start_episode
    solved = False

    def save_state():
        agent.save_state(cfg.state_file, episode=episode, run_best=float(run_best))

    try:
        while episode < cfg.num_episodes:
            state, _ = env.reset(seed=cfg.seed if episode == 0 else None)
            total_reward, score = 0.0, 0.0
            done, first_step = False, True
            losses, qs = [], []
            steps, started = 0, time.perf_counter()

            while not done:
                action = agent.choose_action(state, env)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                agent.store_experience(
                    state, action, reward, next_state, terminated, episode_start=first_step
                )
                first_step = False
                steps += 1

                result = agent.update_q_network()
                if result:
                    losses.append(result[0])
                    qs.append(result[1])

                state = next_state
                total_reward += reward
                score += info.get("raw_reward", reward)

            episode += 1
            steps_per_sec = steps / max(time.perf_counter() - started, 1e-9)
            reward_window.append(score)
            mean_score = float(np.mean(reward_window))

            logger.scalars(
                episode, Reward=total_reward, Score=score, Mean_Score=mean_score,
                Epsilon=agent.epsilon, Steps_Per_Second=steps_per_sec,
                Loss=np.mean(losses) if losses else None,
                Avg_Q_Value=np.mean(qs) if qs else None,
            )
            print(
                f"Ep {episode:5d} | Score: {score:8.1f} | Reward: {total_reward:8.1f} | "
                f"Média({len(reward_window)}): {mean_score:7.1f} | "
                f"ε: {agent.epsilon:.3f} | Buffer: {len(agent.replay_buffer):6d} | "
                f"{steps:4d} passos, {steps_per_sec:6.1f}/s"
            )

            if episode % cfg.eval_interval == 0 and agent.train_step > 0:
                eval_mean, _ = evaluate(eval_env, agent, cfg.eval_episodes, seed=cfg.eval_seed)
                logger.scalars(episode, Eval_Score=eval_mean)
                run_best = max(run_best, eval_mean)
                if eval_mean > file_best:
                    file_best = eval_mean
                    agent.q_network.save_weights(str(cfg.weights_file))
                    _write_meta(cfg, eval_mean, episode)
                    print(f"🏆 Novo melhor modelo (score {eval_mean:.1f}). Pesos salvos.")
                else:
                    print(f"   Não superou o melhor em disco ({file_best:.1f}); pesos mantidos.")
                save_state()

                if cfg.target_score is not None and eval_mean >= cfg.target_score:
                    solved = True
                    print(f"\n✅ Resolvido em {episode} episódios (score greedy {eval_mean:.1f}).")
                    break
    except KeyboardInterrupt:
        print("\n⏹️ Treino interrompido.")
    finally:
        if agent.train_step > 0:
            save_state()
            print(f"💾 Estado do treino salvo em {cfg.state_file.name} (retome com --resume).")
        logger.close()

    if run_best > -np.inf:
        print(f"Melhor avaliação desta execução: {run_best:.1f} | Melhor em disco: {file_best:.1f}")
    else:
        print("Nenhuma avaliação foi feita nesta execução; os pesos em disco não foram alterados.")
    return solved


# =============================================================================
# LINHA DE COMANDO
# =============================================================================
def run_cli(cfg, make_env, build_agent, argv=None):
    """
    Ponto de entrada comum dos scripts.

    make_env(render_mode) -> ambiente Gymnasium
    build_agent(env)      -> DQNAgent novo (redes + buffer)
    """
    parser = argparse.ArgumentParser(description=f"Agente DQN - {cfg.name}")
    parser.add_argument("--play", type=int, nargs="?", const=5, metavar="N",
                        help="não treina: carrega os pesos e joga N episódios sem exploração (padrão 5)")
    parser.add_argument("--render", action="store_true",
                        help="abre a janela do ambiente (equivale a RENDER=human)")
    parser.add_argument("--episodes", type=int, help="número máximo de episódios de treino")
    parser.add_argument("--load", action="store_true",
                        help="começa o treino a partir dos pesos salvos")
    parser.add_argument("--resume", action="store_true",
                        help="retoma o treino do último estado salvo")
    parser.add_argument("--seed", type=int, help="semente aleatória")
    parser.add_argument("--target-score", type=float,
                        help="score médio greedy para encerrar o treino (substitui o padrão do ambiente)")
    args = parser.parse_args(argv)

    if args.seed is not None:
        cfg.seed = args.seed
    if args.episodes is not None:
        cfg.num_episodes = args.episodes
    if args.target_score is not None:
        cfg.target_score = args.target_score
    render_mode = "human" if args.render else get_render_mode()
    set_seed(cfg.seed)

    if args.play is not None:
        env = make_env(render_mode)
        try:
            agent = build_agent(env)
            if not try_load_weights([agent.q_network], cfg.weights_file):
                sys.exit("Sem pesos utilizáveis para jogar. Treine o agente primeiro.")
            evaluate(env, agent, args.play, seed=cfg.seed)
        finally:
            env.close()
        return

    env = make_env(render_mode)
    eval_env = make_env(None)
    try:
        env.action_space.seed(cfg.seed)
        file_best = _existing_best_score(cfg, eval_env, build_agent)
        agent = build_agent(env)
        start_episode, run_best = 0, -np.inf

        if args.resume:
            if not cfg.state_file.exists():
                sys.exit(f"Nenhum estado salvo em {cfg.state_file} para retomar.")
            meta = agent.load_state(cfg.state_file)
            start_episode = int(meta.get("episode", 0))
            run_best = float(meta.get("run_best", -np.inf))
            print(f"▶️ Retomando do episódio {start_episode} (ε={agent.epsilon:.3f}). "
                  f"O replay buffer será preenchido novamente.")
        elif args.load:
            if try_load_weights([agent.q_network, agent.target_network], cfg.weights_file):
                agent.set_epsilon_start(cfg.epsilon_when_loaded)

        train(env, eval_env, agent, cfg, file_best=file_best,
              start_episode=start_episode, run_best=run_best)
    finally:
        env.close()
        eval_env.close()
