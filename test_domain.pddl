(define (domain restore_waterway_no_fuel)

;remove requirements that are not needed
(:requirements :typing :strips :fluents :durative-actions :conditional-effects :preferences)

(:types ;todo: enumerate types and their hierarchy here, e.g. car truck bus - vehicle
    location locatable - object 
    asset debris ship - locatable ;only one debris can be at a waypoint. 
    debris_asset scout_asset ship_salvage_asset - asset ;debris asset removes visible debris. scout asset detects and make visible (to other assets) underwater debris. ship salvage asset salvages ships.
    normal_debris underwater_debris - debris ;normal debris is visible to all assets. underwater debris can only be detected and made visible by scout asset.
    authority ;authority controls whether a location restricted. assets cannot move to locations that are restricted
)

; un-comment following line if constants are needed
;(:constants )

(:predicates ;todo: define predicates here
    (is_location_debris_station ?loc - location) ;check if a location is a debris station to unload debris for debris assets
    (is_location_ship_dock ?loc - location) ;check if a location is a ship dock for ship salvage asset to leave ship in
    (is_location_unrestricted ?loc - location);check if a location is unrestricted for traversal
    (is_underwater_debris_visible ?loc - location);check if underwater debris is visible to debris assets. vacuously true if no underwater debris. 
    (is_asset_not_damaged ?ast - asset);check if asset is not damaged (note that an asset that is out of fuel is not damaged)
    (is_asset_not_moving ?ast - asset);check if asset is not moving
    (is_shipwreck ?shp - ship);check if shipwreck
    (is_debris_asset_not_removing_debris ?ast - debris_asset);check if debris asset is not removing debris
    (is_debris_asset_not_unloading_debris ?ast - debris_asset);check if debris asset is not unloading debris
    (is_scout_asset_not_scouting ?ast - scout_asset); check if scout asset is not scouting location
    (is_ship_salvage_asset_not_docking_ship ?ast - ship_salvage_asset) ;check if ship salvage asset is docking ship
    (is_ship_salvage_asset_not_salvaging_ship ?ast - ship_salvage_asset) ;check if ship salvage asset is not salvaging shipwreck
    (is_ship_salvage_asset_not_towing_ship ?ast - ship_salvage_asset) ;check if ship salvage asset is not towing a salvaged ship
    (at ?obj - locatable ?loc - location);check if a locatable object is at a location
    (on ?obj1 - locatable ?obj2 - locatable);check if a locatable object is on another locatable object
    (is_location_connected ?loc1 - location ?loc2 - location);check if two locations are connected for traversal
    (is_location_not_blocked ?loc1 - location ?loc2 - location) ;check if two location are not blocked by debris for traversal. does not affect scout asset.
    (has_authority_over_location ?aut - authority ?loc - location);check if authority has command over a location
)

(:functions ;todo: define numeric functions here
    (debris_weight ?deb - debris)- number ;check weight of debris [kg]
    (asset_speed ?ast - asset)- number ;check traversal speed of asset [km / h]
    (debris_asset_normal_debris_removal_rate ?ast - debris_asset)- number ;check debris remove rate for normal debris of debris asset [kg / h]
    (debris_asset_underwater_debris_removal_rate ?ast - debris_asset)- number ;check debris remove rate for underwater debris of debris asset [kg / h]
    (debris_asset_debris_unload_rate ?ast - debris_asset)- number ;check the debris unload rate at debris station for debris asset [kg / h]
    (debris_asset_debris_capacity ?ast - debris_asset)- number ;check the maximum weight of debris that the debris asset can carry [kg]
    (debris_asset_debris_weight ?ast - debris_asset)- number ;check the current weight of debris that the debris asset is carrying [kg]
    (scout_asset_scout_time ?ast - scout_asset)- number ;check the time required by scout asset to scout waypoint [h]
    (ship_salvage_asset_dock_time ?ast - ship_salvage_asset) - number ;check the time required by ship salvage asset to dock ship [h]
    (ship_salvage_asset_salvage_time ?ast - ship_salvage_asset) - number ;check the time required by ship salvage asset to salvage shipwreck [h]
    (authority_reply_time ?aut - authority)- number ;check how long does the authority take to make waterway that it is in charge of unrestricted [h]
    (connected_location_distance ?loc1 - location ?loc2 - location)- number ;check the distance between two connected location [km]
)

;define actions here

;move debris asset
(:durative-action move_debris_asset
    :parameters (?loc1 - location ?loc2 - location ?ast - debris_asset)
    :duration (= ?duration (/ (connected_location_distance ?loc1 ?loc2) (asset_speed ?ast))) ;duration based on distance between loc1 and loc2 and ast speed
    :condition (and 
        (at start (and
            (is_asset_not_moving ?ast) ;ast must not be moving at the start
            (at ?ast ?loc1) ;ast must be at loc1 at the start
        ))
        (over all (and
            (is_location_unrestricted ?loc2) ;loc2 must be unrestricted throughout traversal
            (is_underwater_debris_visible ?loc2) ;underwater debris must be visible (vacuously true if no underwater debris) at loc2
            (is_asset_not_damaged ?ast) ;ast must not be damaged throughout traversal
            (is_debris_asset_not_removing_debris ?ast) ;ast cannot be removing debris and moving at the same time
            (is_debris_asset_not_unloading_debris ?ast) ;ast cannot be unloading debris and moving at the same time
            (is_location_connected ?loc1 ?loc2) ;loc1 and loc2 are connected throughout traversal
            (is_location_not_blocked ?loc1 ?loc2) ;traversal from loc1 to loc2 must not be blocked by debris 
        ))
    )
    :effect (and 
        (at start (and 
            (not (at ?ast ?loc1)) ;ast leaves loc1
            (not (is_asset_not_moving ?ast)) ;ast is moving now
        ))
        (at end (and 
            (at ?ast ?loc2) ;ast reaches loc2
            (is_asset_not_moving ?ast) ;ast is not moving now
        ))
    )
)

;debris asset to remove all of normal debris at location. debris asset excess capacity must be >= debris weight  
(:durative-action remove_normal_debris_total
    :parameters (?loc1 - location ?loc2 - location ?deb - normal_debris ?ast - debris_asset)
    :duration (= ?duration (/ (debris_weight ?deb) (debris_asset_normal_debris_removal_rate ?ast))) ;duration based on debris weight and removal rate
    :condition (and 
        (at start (and
            (is_debris_asset_not_removing_debris ?ast) ;ast must not be removing debris at the start
            (at ?deb ?loc1) ;deb must be at loc1
            (at ?deb ?loc2) ;deb must be at loc2
            (> (debris_weight ?deb) 0) ;deb must have positive weight to be removed
            (> (debris_asset_debris_capacity ?ast) (debris_asset_debris_weight ?ast)) ;ast must have capacity to carry more debris
            (>= (- (debris_asset_debris_capacity ?ast) (debris_asset_debris_weight ?ast)) (debris_weight ?deb)) ;ast excess capacity >= deb weight
        ))
        (over all (and
            (is_asset_not_damaged ?ast) ;ast must not be damaged throughout removal
            (is_asset_not_moving ?ast) ;ast cannot be moving and removing debris at the same time
            (is_debris_asset_not_unloading_debris ?ast) ;ast cannot be unloading debris and removing debris at the same time
            (at ?ast ?loc1) ;ast must remain at loc1 throughout removal
            (>= (debris_weight ?deb) 0) ;deb weight cannot be negative throughout removal
            (>= (debris_asset_debris_capacity ?ast) (debris_asset_debris_weight ?ast)) ;ast debris weight cannot exceed its debris capacity
        ))
    )
    :effect (and
        (at start (and 
            (not (is_debris_asset_not_removing_debris ?ast)) ;ast is removing debris now
        )) 
        (at end (and
            (is_debris_asset_not_removing_debris ?ast) ;ast is not removing debris now
            (not (at ?deb ?loc1)) ;deb is not at loc1
            (not (at ?deb ?loc2)) ;deb is not at loc2
            (is_location_not_blocked ?loc1 ?loc2) ;loc2 not blocked from loc1
            (is_location_not_blocked ?loc2 ?loc1) ;loc1 not blocked from loc2
            (increase (debris_asset_debris_weight ?ast) (debris_weight ?deb)) ;load all of deb to ast
            (assign (debris_weight ?deb) 0) ;deb totally removed
        ))
    )
)

;debris asset to remove part of normal debris at location. debris asset must use up all of its remaining capacity.
(:durative-action remove_normal_debris_partial
    :parameters (?loc1 - location ?loc2 - location ?deb - normal_debris ?ast - debris_asset)
    :duration (= ?duration (/ (- (debris_asset_debris_capacity ?ast) (debris_asset_debris_weight ?ast)) (debris_asset_normal_debris_removal_rate ?ast))) 
    :condition (and 
        (at start (and
            (is_debris_asset_not_removing_debris ?ast) ;ast must not be removing debris at the start
            (> (debris_asset_debris_capacity ?ast) (debris_asset_debris_weight ?ast)) ;ast must have capacity to carry more debris
            (> (debris_weight ?deb) (- (debris_asset_debris_capacity ?ast) (debris_asset_debris_weight ?ast))) ;deb weight > ast asset excess capacity
        ))
        (over all (and
            (is_asset_not_damaged ?ast) ;ast must not be damaged throughout removal
            (is_asset_not_moving ?ast) ;ast cannot be moving and removing debris at the same time
            (is_debris_asset_not_unloading_debris ?ast) ;ast cannot be unloading debris and removing debris at the same time
            (at ?ast ?loc1) ;ast must remain at loc1 throughout removal
            (at ?deb ?loc1) ;deb must be at loc1
            (at ?deb ?loc2) ;deb must be at loc2
            (> (debris_weight ?deb) 0) ;deb must have some weight left for partial removal    
            (>= (debris_asset_debris_capacity ?ast) (debris_asset_debris_weight ?ast)) ;ast debris weight cannot exceed its debris capacity
        ))
    )
    :effect (and
        (at start (and 
            (not (is_debris_asset_not_removing_debris ?ast)) ;ast is removing debris now
        )) 
        (at end (and 
            (is_debris_asset_not_removing_debris ?ast) ;ast is not removing debris now
            (decrease (debris_weight ?deb) (- (debris_asset_debris_capacity ?ast) (debris_asset_debris_weight ?ast))) ;deb partially removed
            (assign (debris_asset_debris_weight ?ast) (debris_asset_debris_capacity ?ast)) ;load part of deb to ast
        ))
    )
)

;debris asset to remove all of underwater debris at location. debris asset excess capacity must be >= debris weight  
(:durative-action remove_underwater_debris_total
    :parameters (?loc1 - location ?loc2 - location ?deb - underwater_debris ?ast - debris_asset)
    :duration (= ?duration (/ (debris_weight ?deb) (debris_asset_underwater_debris_removal_rate ?ast))) ;duration based on debris weight and removal rate
    :condition (and 
        (at start (and
            (is_debris_asset_not_removing_debris ?ast) ;ast must not be removing debris at the start
            (at ?deb ?loc1) ;deb must be at loc1
            (at ?deb ?loc2) ;deb must be at loc2
            (> (debris_weight ?deb) 0) ;deb must have positive weight to be removed
            (> (debris_asset_debris_capacity ?ast) (debris_asset_debris_weight ?ast)) ;ast must have capacity to carry more debris
            (>= (- (debris_asset_debris_capacity ?ast) (debris_asset_debris_weight ?ast)) (debris_weight ?deb)) ;ast excess capacity >= deb weight
        ))
        (over all (and
            (is_underwater_debris_visible ?loc1) ;underwater debris must be visible at loc1
            (is_underwater_debris_visible ?loc2) ;underwater debris must be visible at loc2
            (is_asset_not_damaged ?ast) ;ast must not be damaged throughout removal
            (is_asset_not_moving ?ast) ;ast cannot be moving and removing debris at the same time
            (is_debris_asset_not_unloading_debris ?ast) ;ast cannot be unloading debris and removing debris at the same time
            (at ?ast ?loc1) ;ast must remain at loc1 throughout removal
            (>= (debris_weight ?deb) 0) ;deb weight cannot be negative throughout removal
            (>= (debris_asset_debris_capacity ?ast) (debris_asset_debris_weight ?ast)) ;ast debris weight cannot exceed its debris capacity
        ))
    )
    :effect (and
        (at start (and 
            (not (is_debris_asset_not_removing_debris ?ast)) ;ast is removing debris now
        )) 
        (at end (and
            (is_debris_asset_not_removing_debris ?ast) ;ast is not removing debris now
            (not (at ?deb ?loc1)) ;deb is not at loc1
            (not (at ?deb ?loc2)) ;deb is not at loc2
            (is_location_not_blocked ?loc1 ?loc2) ;loc2 not blocked from loc1
            (is_location_not_blocked ?loc2 ?loc1) ;loc1 not blocked from loc2
            (increase (debris_asset_debris_weight ?ast) (debris_weight ?deb)) ;load all of deb to ast
            (assign (debris_weight ?deb) 0) ;deb totally removed
        ))
    )
)

;debris asset to remove part of underwater debris at location. debris asset must use up all of its remaining capacity.
(:durative-action remove_underwater_debris_partial
    :parameters (?loc1 - location ?loc2 - location ?deb - underwater_debris ?ast - debris_asset)
    :duration (= ?duration (/ (- (debris_asset_debris_capacity ?ast) (debris_asset_debris_weight ?ast)) (debris_asset_underwater_debris_removal_rate ?ast))) 
    :condition (and 
        (at start (and
            (is_debris_asset_not_removing_debris ?ast) ;ast must not be removing debris at the start
            (> (debris_asset_debris_capacity ?ast) (debris_asset_debris_weight ?ast)) ;ast must have capacity to carry more debris
            (> (debris_weight ?deb) (- (debris_asset_debris_capacity ?ast) (debris_asset_debris_weight ?ast))) ;deb weight > ast asset excess capacity
        ))
        (over all (and
            (is_underwater_debris_visible ?loc1) ;underwater debris must be visible at loc1
            (is_underwater_debris_visible ?loc2) ;underwater debris must be visible at loc2
            (is_asset_not_damaged ?ast) ;ast must not be damaged throughout removal
            (is_asset_not_moving ?ast) ;ast cannot be moving and removing debris at the same time
            (is_debris_asset_not_unloading_debris ?ast) ;ast cannot be unloading debris and removing debris at the same time
            (at ?ast ?loc1) ;ast must remain at loc1 throughout removal
            (at ?deb ?loc1) ;deb must be at loc1
            (at ?deb ?loc2) ;deb must be at loc2
            (> (debris_weight ?deb) 0) ;deb must have some weight left for partial removal    
            (>= (debris_asset_debris_capacity ?ast) (debris_asset_debris_weight ?ast)) ;ast debris weight cannot exceed its debris capacity
        ))
    )
    :effect (and
        (at start (and 
            (not (is_debris_asset_not_removing_debris ?ast)) ;ast is removing debris now
        )) 
        (at end (and 
            (is_debris_asset_not_removing_debris ?ast) ;ast is not removing debris now
            (decrease (debris_weight ?deb) (- (debris_asset_debris_capacity ?ast) (debris_asset_debris_weight ?ast))) ;deb partially removed
            (assign (debris_asset_debris_weight ?ast) (debris_asset_debris_capacity ?ast)) ;load part of deb to ast
        ))
    )
)

;debris asset to fully unload debris to a debris station
(:durative-action unload_debris_debris_station
    :parameters (?loc - location ?ast - debris_asset)
    :duration (= ?duration (/ (debris_asset_debris_weight ?ast) (debris_asset_debris_unload_rate ?ast))) ;duration based debris weight on ast and unload rate
    :condition (and 
        (at start (and
            (is_debris_asset_not_unloading_debris ?ast) ;ast must not be unloading debris at the start
            (> (debris_asset_debris_weight ?ast) 0) ;ast must have debris to unload 
        ))
        (over all (and
            (is_location_debris_station ?loc) ;loc is debris station
            (is_asset_not_damaged ?ast) ;ast must not be damaged throughout unloading
            (is_asset_not_moving ?ast) ;ast cannot be moving and unloading debris at the same time
            (is_debris_asset_not_removing_debris ?ast) ;ast cannot be removing debris and unloading debris at the same time
            (at ?ast ?loc) ;ast must be at debris station throughout unloading
            (>= (debris_asset_debris_weight ?ast) 0) ;ast debris weight cannot be negative
        ))
    )
    :effect (and
        (at start (and 
            (not (is_debris_asset_not_unloading_debris ?ast)) ;ast is unloading debris now
        ))  
        (at end (and 
            (is_debris_asset_not_unloading_debris ?ast) ;ast is not unloading debris now
            (assign (debris_asset_debris_weight ?ast) 0) ;fully unload debris
        ))
    )
)

;move scout asset
(:durative-action move_scout_asset
    :parameters (?loc1 - location ?loc2 - location ?ast - scout_asset)
    :duration (= ?duration (/ (connected_location_distance ?loc1 ?loc2) (asset_speed ?ast))) ;duration based on distance between loc1 and loc2 and ast speed
    :condition (and 
        (at start (and
            (is_asset_not_moving ?ast) ;ast must not be moving at the start
            (at ?ast ?loc1) ;ast must be at loc1 at the start
        ))
        (over all (and
            (is_location_unrestricted ?loc2) ;loc2 must be unrestricted throughout traversal
            (is_asset_not_damaged ?ast) ;ast must not be damaged throughout traversal
            (is_scout_asset_not_scouting ?ast) ;ast cannot be scouting and moving at the same time
            (is_location_connected ?loc1 ?loc2) ;loc1 and loc2 are connected throughout traversal
        ))
    )
    :effect (and 
        (at start (and 
            (not (at ?ast ?loc1)) ;ast leaves loc1
            (not (is_asset_not_moving ?ast)) ;ast is moving now
        ))
        (at end (and 
            (at ?ast ?loc2) ;ast reaches loc2
            (is_asset_not_moving ?ast) ;ast is not moving now
        ))
    )
)

;scouting of scout asset to locate underwater debris at a location
(:durative-action scout_location
    :parameters (?loc - location ?ast - scout_asset)
    :duration (= ?duration (scout_asset_scout_time ?ast)) ;duration of action is a function of the scout time of the scout asset
    :condition (and
        (at start (and
            (is_scout_asset_not_scouting ?ast) ;ast must not be scouting at the start
        ))
        (over all (and
            (is_asset_not_damaged ?ast) ;ast must not be damaged throughout scouting  
            (is_asset_not_moving ?ast) ;ast cannot be moving and scouting at the same time
            (at ?ast ?loc) ;ast remains at loc for the entire duration of the scouting
        ))
    )
    :effect (and
        (at start (and 
            (not (is_scout_asset_not_scouting ?ast)) ;ast is scouting now
        ))  
        (at end (and
            (is_underwater_debris_visible ?loc) ;underwater debris at location is now visible to all debris assets
            (is_scout_asset_not_scouting ?ast) ;ast is not scouting now
        ))
    )
)

;move ship salvage asset
(:durative-action move_ship_salvage_asset
    :parameters (?loc1 - location ?loc2 - location ?ast - ship_salvage_asset)
    :duration (= ?duration (/ (connected_location_distance ?loc1 ?loc2) (asset_speed ?ast))) ;duration based on distance between loc1 and loc2 and ast speed
    :condition (and 
        (at start (and
            (is_asset_not_moving ?ast) ;ast must not be moving at the start
            (at ?ast ?loc1) ;ast must be at loc1 at the start
        ))
        (over all (and
            (is_location_unrestricted ?loc2) ;loc2 must be unrestricted throughout traversal
            (is_underwater_debris_visible ?loc2) ;underwater debris must be visible (vacuously true if no underwater debris) at loc2
            (is_asset_not_damaged ?ast) ;ast must not be damaged throughout traversal
            (is_ship_salvage_asset_not_docking_ship ?ast) ;ast cannot be docking ship and moving at the same time
            (is_ship_salvage_asset_not_salvaging_ship ?ast) ;ast cannot be salvaging ship and moving at the same time
            (is_location_connected ?loc1 ?loc2) ;loc1 and loc2 are connected throughout traversal
            (is_location_not_blocked ?loc1 ?loc2) ;traversal from loc1 to loc2 must not be blocked by debris 
        ))
    )
    :effect (and 
        (at start (and 
            (not (at ?ast ?loc1)) ;ast leaves loc1
            (not (is_asset_not_moving ?ast)) ;ast is moving now
        ))
        (at end (and 
            (at ?ast ?loc2) ;ast reaches loc2
            (is_asset_not_moving ?ast) ;ast is not moving now
        ))
    )
)

;ship salvage asset salvaging ship
(:durative-action ship_salvage_asset_salvage_ship 
    :parameters (?loc - location ?shp - ship ?ast - ship_salvage_asset)
    :duration (= ?duration (ship_salvage_asset_salvage_time ?ast)) ;duration of action is a function of the ship salvage time of the ship salvage asset
    :condition (and 
        (at start (and
            (is_ship_salvage_asset_not_salvaging_ship ?ast) ;ast cannot be salvaging ship at the start
            (is_ship_salvage_asset_not_towing_ship ?ast) ;ast cannot be towing another salvaged ship at the start
            (at ?shp ?loc) ;shp must be at loc at the start  
        ))
        (over all (and 
            (is_asset_not_damaged ?ast) ;ast must not be damaged during ship salvage
            (is_asset_not_moving ?ast) ;ast cannot be moving and salvaging a ship at the same time
            (is_ship_salvage_asset_not_docking_ship ?ast) ;ast cannot be docking ship and salvaging a ship at the same time
            (is_shipwreck ?shp) ;shp is shipwreck
            (at ?ast ?loc) ;ast remains at loc for the entire duration of ship salvaging
        ))
    )
    :effect (and 
        (at start (and
            (not (is_ship_salvage_asset_not_salvaging_ship ?ast)) ;ast is salvaging shp now
            (not (at ?shp ?loc)) ;shp is not on loc
        ))
        (at end (and
            (is_ship_salvage_asset_not_salvaging_ship ?ast) ;ast is not salvaging shp now
            (not (is_ship_salvage_asset_not_towing_ship ?ast)) ;ast is towing shp now
            (on ?shp ?ast) ;shp is on ast
        ))
    )
)

;ship salvage asset docking ship
(:durative-action ship_salvage_asset_dock_ship
    :parameters (?loc - location ?shp - ship ?ast - ship_salvage_asset)
    :duration (= ?duration (ship_salvage_asset_dock_time ?ast)) ;duration of action is a function of the ship dock time of the ship salvage asset
    :condition (and 
        (at start (and
            (is_ship_salvage_asset_not_docking_ship ?ast) ;ast is not docking ship at the start
            (on ?shp ?ast) ;shp is on ast
        ))
        (over all (and
            (is_location_ship_dock ?loc) ;loc is ship dock
            (is_asset_not_damaged ?ast) ;ast must not be damaged during ship docking
            (is_asset_not_moving ?ast) ;ast cannot be moving and docking a ship at the same time
            (is_ship_salvage_asset_not_salvaging_ship ?ast) ;ast cannot be salvaging a ship and docking ship at the same time
            (is_shipwreck ?shp) ;shp is shipwreck
            (at ?ast ?loc) ;ast remains at loc for the entire duration of ship docking 
        ))
    )
    :effect (and 
        (at start (and
            (not (is_ship_salvage_asset_not_docking_ship ?ast)) ;ast is docking shp now
            (not (on ?shp ?ast)) ;shp is not on ast
        ))
        (at end (and
            (is_ship_salvage_asset_not_docking_ship ?ast) ;ast is not docking shp now
            (is_ship_salvage_asset_not_towing_ship ?ast) ;ast is not towing shp now
            (at ?shp ?loc) ;shp is at ship dock 
        ))
    )
)

;authority to make a location unrestricted 
(:durative-action authority_make_location_unrestricted
    :parameters (?loc - location ?aut - authority)
    :duration (= ?duration (authority_reply_time ?aut)) ;duration is a function of reply time of authority
    :condition (and
        (over all (and
            (has_authority_over_location ?aut ?loc) ;only relevant aut can make loc unrestricted
        ))
    )
    :effect (and 
        (at end (and
            (is_location_unrestricted ?loc) ;loc is unrestricted at the end
        ))
    )
)

)