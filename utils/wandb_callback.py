from stable_baselines3.common.callbacks import BaseCallback
import wandb
import os
import logging

class WandbCallback(BaseCallback):
    def __init__(self, save_freq, save_path):
        """
        :param save_freq: 每多少步保存一次模型
        :param save_path: 保存的目录
        """
        super(WandbCallback, self).__init__()
        self.save_freq = save_freq
        self.save_path = save_path
        self.episode_count = 0  
        self.episode_successes = 0

        # 创建保存路径
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        metrics = {}

        for info in infos:
            if info.get('done', False):
                self.episode_count += 1
                for key, value in info.items():
                    if key.startswith("episode/"):
                        metrics[key] = value

                if info.get('arrival', False):
                    self.episode_successes += 1

                success_rate = self.episode_successes / self.episode_count if self.episode_count > 0 else 0.0
                metrics['episode/success_rate'] = success_rate

        if metrics:
            wandb.log(metrics, step=self.num_timesteps)

        return True

    def _on_rollout_end(self) -> None:
        metrics = {
            'train/policy_loss': self.model.logger.name_to_value.get('train/policy_loss', 0.0),
            'train/value_loss': self.model.logger.name_to_value.get('train/value_loss', 0.0),
            'train/entropy_loss': self.model.logger.name_to_value.get('train/entropy_loss', 0.0),
            'train/approx_kl': self.model.logger.name_to_value.get('train/approx_kl', 0.0),
            'train/clip_fraction': self.model.logger.name_to_value.get('train/clip_fraction', 0.0),
            'rollout/ep_rew_mean': self.model.logger.name_to_value.get('rollout/ep_rew_mean', 0.0),
            'rollout/ep_len_mean': self.model.logger.name_to_value.get('rollout/ep_len_mean', 0.0),
            'time/iterations': self.model.logger.name_to_value.get('time/iterations', 0.0),
            'time/time_elapsed': self.model.logger.name_to_value.get('time/time_elapsed', 0.0),
        }

        wandb.log(metrics, step=self.num_timesteps)

        # 每save_freq步保存一次模型
        if self.num_timesteps % self.save_freq == 0:
            checkpoint_filename = os.path.join(
                self.save_path, f"checkpoint_{self.num_timesteps}_steps.zip"
            )
            self.model.save(checkpoint_filename)
            logging.info(f"已保存 checkpoint: {checkpoint_filename}")
