import logging

from .area import Area


class Board(object):
    """Game board
    """
    def __init__(self, areas, board):
        """
        Parameters
        ----------
        areas : dict of int: list of int
            Dictionary of game areas and their neighbours
        board : dict
            Dictionary describing the game's board
        """
        self.logger = logging.getLogger('CLIENT')
        self.areas = {}
        for area in areas:
            self.areas[area] = Area(area, areas[area]['owner'], areas[area]['dice'],
                                    board[area]['neighbours'], board[area]['hexes'])

    def get_area(self, idx):
        """Get Area given its name
        """
        return self.areas[str(idx)]

    def get_player_areas(self, player_name):
        return [area for area in self.areas.values() if area.get_owner_name() == player_name]

    def get_player_border(self, player_name):
        return [area for area in self.get_player_areas(player_name) if self.is_at_border(area)]

    def get_player_dice(self, player_name):
        """Get all dice of a single player
        """
        return sum([area.get_dice() for area in self.get_player_areas(player_name)])

    def get_players_regions(self, player_name, skip_area=None):
        area_names_to_test = [area.get_name() for area in self.get_player_areas(player_name) if area.get_name() != skip_area]

        if not area_names_to_test:
            return [[]]

        regions = []
        while area_names_to_test:
            area_names_in_current_region = self.get_areas_region(area_names_to_test[0], area_names_to_test)
            assert(len(area_names_in_current_region) > 0)
            regions.append(area_names_in_current_region)

            for area in area_names_in_current_region:
                area_names_to_test.remove(area)

        return regions

    def get_areas_region(self, area_name, available_areas):
        to_test = {area_name}
        current_region = []

        while to_test:
            current_area = to_test.pop()

            if current_area in current_region:
                continue

            current_region.append(current_area)

            for neighbour_name in self.get_area(current_area).get_adjacent_areas():
                if neighbour_name in current_region:
                    continue
                if neighbour_name in current_region:
                    continue

                if neighbour_name in available_areas:
                    to_test.add(neighbour_name)

        return current_region

    def is_at_border(self, area):
        owner = area.get_owner_name()
        neighbourhood_names = area.get_adjacent_areas()

        for neighbour_name in neighbourhood_names:
            neighbour = self.get_area(neighbour_name)
            if neighbour.get_owner_name() != owner:
                return True

        return False

    def nb_players_alive(self):
        return len(set(area.get_owner_name() for area in self.areas.values()))
