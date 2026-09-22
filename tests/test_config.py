import pytest
from yolox.config import YoloxConfig

@pytest.mark.parametrize(
    ('value', 'expected'),
    [
        ('False', False),
        ('false', False),
        ('true', True),
        (' TRUE ', True),
        (' false   ', False),
        (False, False),
        (True, True),
    ],
)
def test_update_boolean_values(value, expected):
    config = YoloxConfig('test')
    config.update({'enable_mixup': value})
    assert config.enable_mixup is expected


@pytest.mark.parametrize(
    'value',
    [
        'yes',
        'no',
        '1',
        '0',
        1,
        0,
        'invalid',
    ],
)
def test_update_rejects_invalid_boolean_values(value):
    config = YoloxConfig('test')

    with pytest.raises(ValueError, match='Invalid boolean value for enable_mixup'):
        config.update({'enable_mixup': value})