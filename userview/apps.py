from django.apps import AppConfig


class UserviewConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'userview'

    def ready(self):
        # Import signals to register cache invalidation hooks
        try:
            import userview.signals  # noqa: F401
        except Exception:
            # Do not raise during startup if signals cannot be imported
            pass
