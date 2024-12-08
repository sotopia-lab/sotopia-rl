# apps.py
from django.apps import AppConfig


class SotopiaConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "sotopia_server"

class RejectionSamplerConfig(AppConfig):
    name = 'sotopia_server'
    rejection_sampler = None  # Initialized in the custom command