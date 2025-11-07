from django.core.management.base import BaseCommand
from django.db import transaction

from userview.models import User


class Command(BaseCommand):
    help = "Clear all users' stored face embeddings (face_data) to re-enroll with new model."

    def add_arguments(self, parser):
        parser.add_argument(
            '--dry-run', action='store_true', help='Show how many records would be cleared without saving.'
        )
        parser.add_argument(
            '--only-empty', action='store_true', help='Only clear corrupted/empty records (no embeddings).'
        )

    def handle(self, *args, **options):
        dry_run = options.get('dry_run', False)
        only_empty = options.get('only_empty', False)

        qs = User.objects.exclude(face_data=None)

        to_clear = []
        for u in qs.iterator():
            fd = u.face_data if isinstance(u.face_data, dict) else None
            if fd is None:
                continue
            if only_empty:
                embs = fd.get('embeddings') if isinstance(fd, dict) else None
                if embs and len(embs) > 0:
                    continue
            to_clear.append(u.id)

        count = len(to_clear)
        if count == 0:
            self.stdout.write(self.style.WARNING('No users matched for clearing.'))
            return

        self.stdout.write(self.style.NOTICE(f'Total users to clear: {count}'))

        if dry_run:
            self.stdout.write(self.style.SUCCESS('Dry run complete. No changes made.'))
            return

        with transaction.atomic():
            for uid in to_clear:
                User.objects.filter(id=uid).update(face_data=None)

        self.stdout.write(self.style.SUCCESS(f'Cleared face_data for {count} users.'))
