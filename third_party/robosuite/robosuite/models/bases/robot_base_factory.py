"""
Defines a string based method of initializing mounts
"""

from typing import Optional

from robosuite.models.bases.robot_base_model import RobotBaseModel


def robot_base_factory(name: Optional[str], idn=0) -> RobotBaseModel:
    """
    Generator for grippers

    Creates a RobotBaseModel instance with the provided name.

    Args:
        name (None or str): the name of the mount class
        idn (int or str): Number or some other unique identification string for this mount instance

    Returns:
        RobotBaseModel class: e.g. MobileBaseModel, LegBaseModel, or MountModel instance

    Raises:
        XMLError: [invalid XML]
    """
    # Import MOUNT_MAPPING at runtime so we avoid circular imports
    from robosuite.models.bases import BASE_MAPPING

    if name not in BASE_MAPPING:
        raise ValueError(f"Unknown base name: '{name}'. Available bases: {list(BASE_MAPPING.keys())}")

    return BASE_MAPPING[name](idn=idn)
