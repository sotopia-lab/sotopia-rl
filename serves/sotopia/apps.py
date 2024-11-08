# apps.py
from django.apps import AppConfig


class SotopiaConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "sotopia"

class RejectionSamplerConfig(AppConfig):
    name = 'sotopia'
    rejection_sampler = None  # Initialized in the custom command
