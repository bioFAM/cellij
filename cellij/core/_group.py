class Group:
    """Top-level object to collect features or observations.


    Attributes
    ----------
    name : str
        name of the group
    members : list
        list of features or observations belonging to the group
    level : str
        level of the group, either 'feature' or 'obs'

    """

    def __init__(self, name, members, level):
        self._name = name
        self._members = members
        self._level = level
