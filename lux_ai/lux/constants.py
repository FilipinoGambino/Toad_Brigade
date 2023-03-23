class Constants:

    # noinspection PyPep8Naming
    class INPUT_CONSTANTS:
        RESEARCH_POINTS = "rp"
        RESOURCES = "r"
        UNITS = "u"
        CITY = "c"
        CITY_TILES = "ct"
        ROADS = "ccd"
        DONE = "D_DONE"

    class DIRECTIONS:
        CENTER = 0
        UP = 1
        RIGHT = 2
        DOWN = 3
        LEFT = 4

        @staticmethod
        def astuple(include_center: bool):
            move_directions = (
                Constants.DIRECTIONS.UP,
                Constants.DIRECTIONS.DOWN,
                Constants.DIRECTIONS.LEFT,
                Constants.DIRECTIONS.RIGHT,
            )
            if include_center:
                return move_directions + (Constants.DIRECTIONS.CENTER,)
            else:
                return move_directions

    # noinspection PyPep8Naming
    class UNIT_TYPES:
        LIGHT = 0
        HEAVY = 1

    # noinspection PyPep8Naming
    class RESOURCE_TYPES:
        ICE = 1
        WATER = 2
        ORE = 3
        METAL = 4

        @staticmethod
        def astuple():
            return (
                Constants.RESOURCE_TYPES.ICE,
                Constants.RESOURCE_TYPES.WATER,
                Constants.RESOURCE_TYPES.ORE,
                Constants.RESOURCE_TYPES.METAL,
            )
