from django.core.management.base import BaseCommand
from django.db import transaction, models
from django.utils import timezone
from collections import defaultdict

from userview.models import Attendance


class Command(BaseCommand):
    help = "Backfill Attendance.session_day from timestamp and resolve duplicates for unique (student, class_instance, session_day)."

    def add_arguments(self, parser):
        parser.add_argument('--dry-run', action='store_true', help='Do not write changes, only report.')
        parser.add_argument('--batch', type=int, default=1000, help='Batch size for updates.')

    def handle(self, *args, **options):
        dry = options['dry_run']
        batch = options['batch']

        self.stdout.write(self.style.NOTICE('Starting backfill of session_day...'))

        qs = Attendance.objects.filter(session_day__isnull=True)
        total_missing = qs.count()
        self.stdout.write(f"Missing session_day rows: {total_missing}")

        updated = 0
        deleted_dupes = 0

        # Phase 1: populate session_day
        idx = 0
        while idx < total_missing:
            chunk = list(qs.order_by('id')[idx:idx+batch])
            if not chunk:
                break
            for r in chunk:
                ts = r.timestamp or timezone.now()
                local_ts = timezone.localtime(ts) if timezone.is_aware(ts) else ts
                r.session_day = local_ts.date()
            if not dry:
                Attendance.objects.bulk_update(chunk, ['session_day'], batch_size=batch)
            updated += len(chunk)
            idx += batch
            self.stdout.write(f"Backfilled {updated}/{total_missing}")

        # Phase 2: resolve duplicates by (student, class_instance, session_day)
        self.stdout.write(self.style.NOTICE('Resolving duplicates...'))
        # Build groups of duplicates
        dupes_map = defaultdict(list)
        for row in Attendance.objects.values('student_id', 'class_instance_id', 'session_day').annotate(c=models.Count('id')).filter(c__gt=1):
            dupes_map[(row['student_id'], row['class_instance_id'], row['session_day'])] = []
        if dupes_map:
            # Load records for the duplicate keys
            keys = list(dupes_map.keys())
            for (sid, cid, day) in keys:
                dupes = list(Attendance.objects.filter(student_id=sid, class_instance_id=cid, session_day=day).order_by('id'))
                dupes_map[(sid, cid, day)] = dupes

            with transaction.atomic():
                for key, records in dupes_map.items():
                    if len(records) <= 1:
                        continue
                    # Keep the earliest, consolidate presence: if any present -> mark kept as present
                    any_present = any(r.is_present for r in records)
                    keep = records[0]
                    if not dry:
                        if keep.is_present != any_present:
                            keep.is_present = any_present
                            keep.save(update_fields=['is_present'])
                        # delete the rest
                        to_delete = [r.id for r in records[1:]]
                        Attendance.objects.filter(id__in=to_delete).delete()
                    deleted_dupes += (len(records) - 1)

        self.stdout.write(self.style.SUCCESS(
            f"Done. Backfilled={updated}, Duplicates removed={deleted_dupes}, DryRun={dry}"
        ))
