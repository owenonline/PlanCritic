# Domain: restore_waterway_no_fuel

The following gives a brief summary of the key details of the domain (restore_waterway_no_fuel) and objectives of the task.

## Objects

```pddl
(:types
    location locatable - object
    asset debris ship - locatable
    debris_asset scout_asset ship_salvage_asset - asset
    normal_debris underwater_debris - debris
    authority
)
```

Brief summary of objects:

- `asset`:
  - `debris_asset`: Removes visible debris from waterway by loading them on itself
  - `scout_asset`: Detects and makes visible to debris assets the underwater debris at a location. The logic behind this in reality is that the `underwater_debris` is localised to a certain degree with uncertainty and the `scout_asset` is required to precisely localise it in order for the `debris_asset` to safely traverse to the location and safely remove it.
  - `ship_salvage_asset`: Salvages a ship that is shipwrecked and tows it back to a ship dock.
- `debris`:
  - `normal_debris`: Visible to all assets
  - `underwater_debris`: Can only be detected and made visible by `scout_asset`
- `ship`: Ship on a waterway that could be shipwreck. Can be loaded on ship salvage asset to be towed.
- `authority`: Controls whether a location is restricted. Assets cannot move to locations that are restricted.

## Predicates

```pddl
(:predicates
    (is_location_debris_station ?loc - location)
    (is_location_ship_dock ?loc - location)
    (is_location_unrestricted ?loc - location)
    (is_underwater_debris_visible ?loc - location)
    (is_asset_not_damaged ?ast - asset)
    (is_asset_not_moving ?ast - asset)
    (is_shipwreck ?shp - ship)
    (is_debris_asset_not_removing_debris ?ast - debris_asset)
    (is_debris_asset_not_unloading_debris ?ast - debris_asset)
    (is_scout_asset_not_scouting ?ast - scout_asset)
    (is_ship_salvage_asset_not_docking_ship ?ast - ship_salvage_asset)
    (is_ship_salvage_asset_not_salvaging_ship ?ast - ship_salvage_asset)
    (is_ship_salvage_asset_not_towing_ship ?ast - ship_salvage_asset)
    (at ?obj - locatable ?loc - location)
    (on ?obj1 - locatable ?obj2 - locatable)
    (is_location_connected ?loc1 - location ?loc2 - location)
    (is_location_not_blocked ?loc1 - location ?loc2 - location)
    (has_authority_over_location ?aut - authority ?loc - location)
)
```

Brief summary of predicates:

- `(is_location_debris_station ?loc - location)`: Check if a location is a debris station to unload debris for debris assets
- `(is_location_ship_dock ?loc - location)`: Check if a location is a ship dock for ship salvage asset to leave ship in
- `(is_location_unrestricted ?loc - location)`: Check if a location is unrestricted for traversal
- `(is_underwater_debris_visible ?loc - location)`: Check if underwater debris is visible to debris assets. vacuously true if no underwater debris.
- `(is_asset_not_damaged ?ast - asset)`: Check if asset is not damaged (note that an asset that is out of fuel is not damaged)
- `(is_asset_not_moving ?ast - asset)`: Check if asset is not moving
- `(is_shipwreck ?shp - ship)`: Check if shipwreck
- `(is_debris_asset_not_removing_debris ?ast - debris_asset)`: Check if debris asset is not removing debris
- `(is_debris_asset_not_unloading_debris ?ast - debris_asset)`: Check if debris asset is not unloading debris
- `(is_scout_asset_not_scouting ?ast - scout_asset)`: Check if scout asset is not scouting location
- `(is_ship_salvage_asset_not_docking_ship ?ast - ship_salvage_asset)`: Check if ship salvage asset is docking ship
- `(is_ship_salvage_asset_not_salvaging_ship ?ast - ship_salvage_asset)`: Check if ship salvage asset is not salvaging shipwreck
- `(is_ship_salvage_asset_not_towing_ship ?ast - ship_salvage_asset)`: Check if ship salvage asset is not towing a salvaged ship
- `(at ?obj - locatable ?loc - location)`: Check if a locatable object is at a location
- `(on ?obj1 - locatable ?obj2 - locatable)`: Check if a locatable object is on another locatable object
- `(is_location_connected ?loc1 - location ?loc2 - location)`: Check if two locations are connected for traversal
- `(is_location_not_blocked ?loc1 - location ?loc2 - location)`: Check if two location are not blocked by debris for traversal. does not affect scout asset.
- `(has_authority_over_location ?aut - authority ?loc - location)`: Check if authority has command over a location

## Functions

```pddl
(:functions
    (debris_weight ?deb - debris) - number
    (asset_speed ?ast - asset) - number
    (debris_asset_normal_debris_removal_rate ?ast - debris_asset) - number
    (debris_asset_underwater_debris_removal_rate ?ast - debris_asset) - number
    (debris_asset_debris_unload_rate ?ast - debris_asset) - number
    (debris_asset_debris_capacity ?ast - debris_asset) - number
    (debris_asset_debris_weight ?ast - debris_asset) - number
    (scout_asset_scout_time ?ast - scout_asset) - number
    (ship_salvage_asset_dock_time ?ast - ship_salvage_asset) - number
    (ship_salvage_asset_salvage_time ?ast - ship_salvage_asset) - number
    (authority_reply_time ?aut - authority) - number
    (connected_location_distance ?loc1 - location ?loc2 - location) - number
)
```

Brief summary of functions:

- `(debris_weight ?deb - debris)`: Check weight of debris [kg]
- `(asset_speed ?ast - asset)`: Check traversal speed of asset [km / h]
- `(debris_asset_normal_debris_removal_rate ?ast - debris_asset)`: Check debris remove rate for normal debris of debris asset [kg / h]
- `(debris_asset_underwater_debris_removal_rate ?ast - debris_asset)`: Check debris remove rate for underwater debris of debris asset [kg / h]
- `(debris_asset_debris_unload_rate ?ast - debris_asset)`: Check the debris unload rate at debris station for debris asset [kg / h]
- `(debris_asset_debris_capacity ?ast - debris_asset)`: Check the maximum weight of debris that the debris asset can carry [kg]
- `(debris_asset_debris_weight ?ast - debris_asset)`: Check the current weight of debris that the debris asset is carrying [kg]
- `(scout_asset_scout_time ?ast - scout_asset)`: Check the time required by scout asset to scout waypoint [h]
- `(ship_salvage_asset_dock_time ?ast - ship_salvage_asset)`: Check the time required by ship salvage asset to dock ship [h]
- `(ship_salvage_asset_salvage_time ?ast - ship_salvage_asset)`: Check the time required by ship salvage asset to salvage shipwreck [h]
- `(authority_reply_time ?aut - authority)`: Check how long does the authority take to make waterway that it is in charge of unrestricted [h]
- `(connected_location_distance ?loc1 - location ?loc2 - location)`: Check the distance between two connected location [km]

## Durative Actions

Brief summary of durative actions:

```pddl
(:durative-action move_debris_asset
    :parameters (?loc1 - location ?loc2 - location ?ast - debris_asset)
    :duration (= ?duration (/ (connected_location_distance ?loc1 ?loc2) (asset_speed ?ast)))
```

- `move_debris_asset`: Move `debris_asset` from one `location` to another. `duration` depends on `(connected_location_distance ?loc1 - location ?loc2 - location)` and `(asset_speed ?ast - asset)`.

```pddl
(:durative-action remove_normal_debris_total
    :parameters (?loc1 - location ?loc2 - location ?deb - normal_debris ?ast - debris_asset)
    :duration (= ?duration (/ (debris_weight ?deb) (debris_asset_normal_debris_removal_rate ?ast)))
```

- `remove_normal_debris_total`: `debris_asset` to remove all of `normal_debris` between `loc1` and `loc2`. `debris_asset` excess capacity must be >= `(debris_weight ?deb)`. `duration` depends on `(debris_weight ?deb)` and `(debris_asset_normal_debris_removal_rate ?ast - debris_asset)`.

```pddl
(:durative-action remove_normal_debris_partial
    :parameters (?loc1 - location ?loc2 - location ?deb - normal_debris ?ast - debris_asset)
    :duration (= ?duration (/ (- (debris_asset_debris_capacity ?ast) (debris_asset_debris_weight ?ast)) (debris_asset_normal_debris_removal_rate ?ast)))
```

- `remove_normal_debris_partial`: `debris_asset` to remove part of `normal_debris` between `loc1` and `loc2`. `(debris_weight ?deb)` must be > `debris_asset` excess capacity. `debris_asset` must use up all of its remaining capacity. `duration` depends on `debris_asset` excess capacity and `(debris_asset_normal_debris_removal_rate ?ast - debris_asset)`.

```pddl
(:durative-action remove_underwater_debris_total
    :parameters (?loc1 - location ?loc2 - location ?deb - underwater_debris ?ast - debris_asset)
    :duration (= ?duration (/ (debris_weight ?deb) (debris_asset_underwater_debris_removal_rate ?ast)))
```

- `remove_underwater_debris_total`: `debris_asset` to remove all of `underwater_debris` between `loc1` and `loc2`. `debris_asset` excess capacity must be >= `(debris_weight ?deb)`. `duration` depends on `(debris_weight ?deb)` and `(debris_asset_underwater_debris_removal_rate ?ast - debris_asset)`.

```pddl
(:durative-action remove_underwater_debris_partial
    :parameters (?loc1 - location ?loc2 - location ?deb - underwater_debris ?ast - debris_asset)
    :duration (= ?duration (/ (- (debris_asset_debris_capacity ?ast) (debris_asset_debris_weight ?ast)) (debris_asset_underwater_debris_removal_rate ?ast)))
```

- `remove_underwater_debris_partial`: `debris_asset` to remove part of `underwater_debris` between `loc1` and `loc2`. `(debris_weight ?deb)` must be > `debris_asset` excess capacity. `debris_asset` must use up all of its remaining capacity. `duration` depends on `debris_asset` excess capacity and `(debris_asset_underwater_debris_removal_rate ?ast - debris_asset)`.

```pddl
(:durative-action unload_debris_debris_station
    :parameters (?loc - debris_station ?ast - debris_asset)
    :duration (= ?duration (/ (debris_asset_debris_weight ?ast) (debris_asset_debris_unload_rate ?ast)))
```

- `unload_debris_debris_station`: `debris_asset` to unload all debris to a `debris_station`. `duration` depends on `(debris_asset_debris_weight ?ast - debris_asset)` and `(debris_asset_debris_unload_rate ?ast - debris_asset)`.

```pddl
(:durative-action move_scout_asset
    :parameters (?loc1 - location ?loc2 - location ?ast - scout_asset)
    :duration (= ?duration (/ (connected_location_distance ?loc1 ?loc2) (asset_speed ?ast)))
```

- `move_scout_asset`: Move `scout_asset` from one `location` to another. `duration` depends on `(connected_location_distance ?loc1 - location ?loc2 - location)` and `(asset_speed ?ast - asset)`.

```pddl
(:durative-action scout_location
    :parameters (?loc - location ?ast - scout_asset)
    :duration (= ?duration (scout_asset_scout_time ?ast))
```

- `scout_location`: Scouting of `scout_asset` to locate `underwater_debris` at a `location`. `duration` depends on `(scout_asset_scout_time ?ast - scout_asset)`.

```pddl
(:durative-action move_ship_salvage_asset
    :parameters (?loc1 - location ?loc2 - location ?ast - ship_salvage_asset)
    :duration (= ?duration (/ (connected_location_distance ?loc1 ?loc2) (asset_speed ?ast)))
```

- `move_ship_salvage_asset`: Move `ship_salvage_asset` from one `location` to another. `duration` depends on `(connected_location_distance ?loc1 - location ?loc2 - location)` and `(asset_speed ?ast - asset)`.

```pddl
(:durative-action ship_salvage_asset_salvage_ship
    :parameters (?loc - location ?shp - ship ?ast - ship_salvage_asset)
    :duration (= ?duration (ship_salvage_asset_salvage_time ?ast))
```

- `ship_salvage_asset_salvage_ship`: `ship_salvage_asset` to salvage `ship` that is shipwrecked by loading the `ship` on itself to be towed to a ship dock. `duration` depends on `(ship_salvage_asset_salvage_time ?ast)`.

```pddl
(:durative-action ship_salvage_asset_dock_ship
    :parameters (?loc - location ?shp - ship ?ast - ship_salvage_asset)
    :duration (= ?duration (ship_salvage_asset_dock_time ?ast))
```

- `ship_salvage_asset_dock_ship`: `ship_salvage_asset` to dock `ship` that is shipwrecked at a ship dock. `duration` depends on `(ship_salvage_asset_dock_time ?ast)`.

```pddl
(:durative-action authority_make_location_unrestricted
    :parameters (?loc - location ?aut - authority)
    :duration (= ?duration (authority_reply_time ?aut))
```

- `authority_make_location_unrestricted`: `authority` making a `location` unrestricted. `duration` depends on `(authority_reply_time ?aut - authority)`.

Please refer to the domain files to understand more about the conditions and effects of the durative actions.

## Key Domain Features

The following are some key domains features to be noted. Please refer to the domain file for further details.

- `debris` lies in the channel (edge) connecting two `location` (i.e. `(is_location_connected ?loc1 - location ?loc2 - location)` is True bidirectionally). Hence debris is `at` two `location` in implementation and a location can have multiple `debris` by virtue of multiple channels with debris being connected to it.
- When the debris lying between two connected `location` (i.e. `(is_location_connected ?loc1 - location ?loc2 - location)` is True bidirectionally) is removed, the connected `location` are not blocked bidirectionally (i.e. `(is_location_not_blocked ?loc1 - location ?loc2 - location)` is True bidirectionally)
- `asset` can only do one action at the same time (e.g. cannot move and remove debris at the same time)
- `debris_asset` and `ship_salvage_asset` cannot travel between two connected `location` (i.e. `(is_location_connected ?loc1 - location ?loc2 - location)` is True bidirectionally) that is blocked by debris between them (i.e. `(is_location_not_blocked ?loc1 - location ?loc2 - location)` is False bidirectionally)
- `debris_asset` and `ship_salvage_asset` cannot travel to a `location` where `underwater_debris` is not visible (i.e. `(is_underwater_debris_visible ?loc - location)` is False)
- `debris_asset` cannot remove `underwater_debris` between two `location` where `underwater_debris` is not visible (i.e. `(is_underwater_debris_visible ?loc - location)` is False for both `location`)
- `ship_salvage_asset` can only tow a single ship at a time
- `asset` have infinite fuel

## Goals

- Unblock key waterways by removing associated debris connected to it
- Salvage and tow sunken ship on key waterways to ship dock
- Achieve above while minimising makespan
