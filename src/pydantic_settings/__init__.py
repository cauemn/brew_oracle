import os


class BaseSettings:
    def __init__(self):
        for name, value in self.__class__.__dict__.items():
            if name.isupper():
                env_val = os.getenv(name)
                if env_val is not None:
                    setattr(self, name, env_val)
                else:
                    setattr(self, name, value)


def SettingsConfigDict(*args, **kwargs):
    return dict(*args, **kwargs)
