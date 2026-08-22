import pytest

def test_imports():
    """Проверяем, что все функции импортируются корректно."""
    from gdn2_package import gdn2_pallas_forward_trainable, gdn2_pallas_forward
    assert gdn2_pallas_forward_trainable is not None
    assert gdn2_pallas_forward is not None

def test_version():
    """Проверяем, что версия определена."""
    import gdn2_package
    assert hasattr(gdn2_package, "__version__")
