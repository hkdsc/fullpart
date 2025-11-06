from .personal_configs_part import personal_configs_part
from .personal_configs_part_stage2 import personal_configs_part_s2


train_configs = {
    **personal_configs_part,
    **personal_configs_part_s2,
}
