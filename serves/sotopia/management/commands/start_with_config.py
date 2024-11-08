# sotopia/management/commands/start_with_config.py
from django.core.management import call_command
from django.core.management.base import BaseCommand
from sotopia.apps import RejectionSamplerConfig
from sotopia.models import RejectionSampler


class Command(BaseCommand):
    help = 'Start the server with custom RejectionSampler configuration'

    def add_arguments(self, parser):
        parser.add_argument('--sft_model_path', required=True, type=str, help='Path to the SFT model')
        parser.add_argument('--reward_model_path', required=True, type=str, help='Path to the Reward model')
        parser.add_argument('--model_name', required=True, type=str, help='Name of the model')
        parser.add_argument('--template_path', required=True, type=str, help='Path to the Jinja template')
        parser.add_argument('--max_responses', type=int, default=5, help='Max responses')
        parser.add_argument('--max_length', type=int, default=4096, help='Max length of responses')
        parser.add_argument('--port', type=int, default=8000, help='Port number for the Django server')

    def handle(self, *args, **options):
        # Set up the rejection sampler with the provided config
        config = {
            "sft_model_path": options['sft_model_path'],
            "reward_model_path": options['reward_model_path'],
            "model_name": options['model_name'],
            "template_path": options['template_path'],
            "max_responses": options['max_responses'],
            "max_length": options['max_length'],
        }

        # Initialize the rejection_sampler directly
        RejectionSamplerConfig.rejection_sampler = RejectionSampler(**config)

        # Start the server with the specified port
        self.stdout.write(f"Starting the Django server on port {options['port']} with custom configuration...")
        call_command('runserver', f"{options['port']}")
