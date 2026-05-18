import pytest 

from mrtunneling.task_driver import TaskDriver


class BTD(TaskDriver):
    pass

class BTD2(TaskDriver):
    def energy(self):
        return 0.0

@pytest.mark.parametrize("btd", [BTD, BTD2])
def test_bad_task_driver(btd):
    with pytest.raises(TypeError):
        b = btd()
