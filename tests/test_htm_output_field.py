from psu_capstone.agent_layer.HTM import OutputField

"""++++++++++Output Field Testing++++++++++"""


def test_outputfield_creation():
    o = OutputField(size=16, motor_action=(None,))
    assert isinstance(o, OutputField)


# this is mainly here for a placeholder as I think more will be added to that code.
