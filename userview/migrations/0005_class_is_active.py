from django.db import migrations, models


def backfill_is_active(apps, schema_editor):
    Class = apps.get_model('userview', 'Class')
    # Iterate and set is_active=True for documents missing the field or set to None
    for cls in Class.objects.all():
        try:
            # getattr default handles missing attribute gracefully
            if getattr(cls, 'is_active', None) in (None, ''):
                cls.is_active = True
                cls.save(update_fields=['is_active'])
        except Exception:
            # Best-effort backfill; continue on individual failures
            continue


def noop_forward(apps, schema_editor):
    pass


def noop_reverse(apps, schema_editor):
    pass


class Migration(migrations.Migration):

    dependencies = [
        ('userview', '0004_auto_20251115_2214'),
    ]

    operations = [
        migrations.AddField(
            model_name='class',
            name='is_active',
            field=models.BooleanField(default=True, db_index=True),
        ),
        migrations.RunPython(backfill_is_active, reverse_code=noop_reverse),
    ]
