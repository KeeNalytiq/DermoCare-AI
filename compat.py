import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable


def _make_cast_fallback_class(short_name):
    @register_keras_serializable(package="compat")
    class CastToFloat32Fallback(tf.keras.layers.Layer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def call(self, inputs):
            return tf.cast(inputs, tf.float32)

    CastToFloat32Fallback.__name__ = short_name
    return CastToFloat32Fallback


def get_fallbacks_for_unknown_layer(serialized_name: str):
    """Return a dict suitable for passing as `custom_objects` to Keras load_model.

    The dict maps both the full serialized name (e.g. 'Custom>CastToFloat32') and
    the simple class name ('CastToFloat32') to a minimal fallback layer that
    casts inputs to float32. This preserves behavior for simple casting layers.
    """
    short = serialized_name.split('>')[-1]
    cls = _make_cast_fallback_class(short)
    return {serialized_name: cls, short: cls}
