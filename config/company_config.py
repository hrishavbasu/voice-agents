"""
Active company configuration — currently pointing to Apollo Hospitals Navi Mumbai.

All modules import COMPANY_CONFIG from here. To switch clinics/companies,
update the import below to point to a different config module.
"""

from config.clinic_config import COMPANY_CONFIG  # noqa: F401
