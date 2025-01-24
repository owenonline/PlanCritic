import copy


MAP_CONFIG = {
    "locations": {
        "wpt_ini": [45.717599, -73.440599],
        "wpt_a_0": [45.463874, -73.971181],
        "wpt_b_0": [45.430800, -73.544315],
        "wpt_end": [45.335075, -73.936037],
        "shp_dck_0": [45.761269, -73.390026],
        "deb_stn_0": [45.761269, -73.390026]
    },
    "assets": {
        "shp_0": "ship",
        "deb_ast_0": "debris_asset",
        "sct_ast_0": "scout_asset",
        "shp_sal_ast_0": "ship_salvage_asset"
    },
    "starting_state": {
            "shp_0": "wpt_end",
            "deb_ast_0": "wpt_ini",
            "sct_ast_0": "wpt_ini",
            "shp_sal_ast_0": "wpt_ini"
        }
}

class MapHandler:
    def __init__(self):
        self.current_state: dict = copy.deepcopy(MAP_CONFIG["starting_state"])

    def __call__(self, predicates):
        
        # update the current state
        for predicate in predicates:
            predicate = predicate.replace("(", "").replace(")", "")
            if any([action in predicate for action in ["move_scout_asset", "move_debris_asset", "move_ship_salvage_asset"]]):
                loc1, loc2, asset = predicate.split(" ")[1:]
                self.current_state[asset] = loc2
            elif "ship_salvage_asset_salvage_ship" in predicate:
                loc, ship, salvage_asset = predicate.split(" ")[1:]
                self.current_state.pop(ship) # ship is being towed, so remove the marker
            elif "ship_salvage_asset_dock_ship" in predicate:
                loc, ship, salvage_asset = predicate.split(" ")[1:]
                self.current_state[ship] = loc # ship is being docked, so put the marker back

        # return the format the map is expecting:
        return [
            [MAP_CONFIG["assets"][asset], MAP_CONFIG["locations"][location]]
            for asset, location in self.current_state.items()
        ]

    