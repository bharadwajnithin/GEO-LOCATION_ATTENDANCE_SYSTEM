from django.db.models.signals import post_save, post_delete
from django.dispatch import receiver
from django.core.cache import cache
from django.utils import timezone

from userview.models import Attendance


def _bump_stamp(instance: Attendance):
    try:
        cls_id = instance.class_instance_id
        day = instance.session_day
        if not day:
            ts = instance.timestamp or timezone.now()
            local_ts = timezone.localtime(ts) if timezone.is_aware(ts) else ts
            day = local_ts.date()
        key = f"analytics:stamp:{cls_id}:{day}"
        try:
            cache.incr(key)
        except Exception:
            # If key doesn't exist or backend lacks incr, set to 1 with short TTL
            val = (cache.get(key) or 0) + 1
            cache.set(key, val, timeout=60*60*6)  # 6 hours default for stamps
    except Exception:
        # Never break request flow due to cache issues
        pass


@receiver(post_save, sender=Attendance)
def attendance_saved(sender, instance: Attendance, **kwargs):
    _bump_stamp(instance)


@receiver(post_delete, sender=Attendance)
def attendance_deleted(sender, instance: Attendance, **kwargs):
    _bump_stamp(instance)
