from django.apps import AppConfig

from .models import RejectionSampler


class SotopiaConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "sotopia"

class RejectionSamplerConfig(AppConfig):
    name = 'sotopia'
    rejection_sampler = None

    def ready(self):
        # Load the model once and ensure it is on the GPU
        self.rejection_sampler = RejectionSampler(
            sft_model_path="/data/haofeiy2/sotopia-rl/saves/sft/checkpoint-4000",
            reward_model_path="/data/haofeiy2/sotopia-rl/saves/rm_baseline/checkpoint-14000",
            model_name="/data/haofeiy2/gemma-2-2b-it/",
            template_path="/data/haofeiy2/sotopia-rl/evals/gemma-2-2b-it.jinja",
        )
