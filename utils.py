import argparse

class YamlNamespace(argparse.Namespace):
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
                setattr(self, a, [YamlNamespace(x) if isinstance(x, dict) else x for x in b])
            else:
                setattr(self, a, YamlNamespace(b) if isinstance(b, dict) else b)
