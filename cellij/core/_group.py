class Group:
    """Top-level object to collect features or observations.


    Attributes
    ----------
    name : str
        Name of the group
    members : list
        List of features or observations belonging to the group
    level : str
        Level of the group, either 'feature' or 'obs'

    """

    def __init__(self, name, members, level):
        self._name = name
        self._members = members
        self._level = level
