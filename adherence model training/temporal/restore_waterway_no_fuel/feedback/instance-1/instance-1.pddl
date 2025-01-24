(define (problem deb_sct_shp_sal_ast_1_1_1_c_2_l_1_end_1_problem_1) (:domain restore_waterway_no_fuel)
(:objects
    wpt_ini wpt_end
    wpt_a_0 wpt_b_0
    deb_stn_0
    shp_dck_0 - location
    shp_0 - ship 
    deb_ast_0 - debris_asset
    sct_ast_0 - scout_asset
    shp_sal_ast_0 - ship_salvage_asset
    n_deb_ini_a_0 n_deb_a_0_end - normal_debris
    u_deb_ini_b_0 u_deb_b_0_end - underwater_debris
    aut_a aut_b - authority
)

(:init
    ;define locations predicates / functions
    ;debris station (is_location_debris_station)
    (is_location_debris_station deb_stn_0)
    ;ship dock (is_location_ship_dock)
    (is_location_ship_dock shp_dck_0) 
    ;unrestricted (is_location_unrestricted)
    (is_location_unrestricted deb_stn_0)
    (is_location_unrestricted shp_dck_0)
    (is_location_unrestricted wpt_ini)
    (is_location_unrestricted wpt_end)
    ;underwater debris visible (is_underwater_debris_visible)
    (is_underwater_debris_visible deb_stn_0)
    (is_underwater_debris_visible shp_dck_0)
    (is_underwater_debris_visible wpt_ini)
    ;connectivity (is_location_connected)
    (is_location_connected deb_stn_0 wpt_ini)
    (is_location_connected wpt_ini deb_stn_0)
    (is_location_connected shp_dck_0 wpt_ini)
    (is_location_connected wpt_ini shp_dck_0)
    (is_location_connected wpt_ini wpt_a_0)
    (is_location_connected wpt_a_0 wpt_ini)
    (is_location_connected wpt_a_0 wpt_end)
    (is_location_connected wpt_end wpt_a_0)
    (is_location_connected wpt_ini wpt_b_0)
    (is_location_connected wpt_b_0 wpt_ini)
    (is_location_connected wpt_b_0 wpt_end)
    (is_location_connected wpt_end wpt_b_0)
    ;blocked (is_location_not_blocked)
    (is_location_not_blocked deb_stn_0 wpt_ini)
    (is_location_not_blocked wpt_ini deb_stn_0)
    (is_location_not_blocked shp_dck_0 wpt_ini)
    (is_location_not_blocked wpt_ini shp_dck_0)
    ;distance [km] (connected_location_distance)
    (= (connected_location_distance deb_stn_0 wpt_ini) 10)
    (= (connected_location_distance wpt_ini deb_stn_0) 10)
    (= (connected_location_distance shp_dck_0 wpt_ini) 10)
    (= (connected_location_distance wpt_ini shp_dck_0) 10)
    (= (connected_location_distance wpt_ini wpt_a_0) 10)
    (= (connected_location_distance wpt_a_0 wpt_ini) 10)
    (= (connected_location_distance wpt_a_0 wpt_end) 10)
    (= (connected_location_distance wpt_end wpt_a_0) 10)
    (= (connected_location_distance wpt_ini wpt_b_0) 10)
    (= (connected_location_distance wpt_b_0 wpt_ini) 10)
    (= (connected_location_distance wpt_b_0 wpt_end) 10)
    (= (connected_location_distance wpt_end wpt_b_0) 10)

    ;define ship predicates / functions
    ;shipwreck (is_shipwreck)
    (is_shipwreck shp_0)
    ;location (at)
    (at shp_0 wpt_end)

    ;define asset predicates / functions
    ;common predicates
    ;not damaged (is_asset_not_damaged)
    (is_asset_not_damaged deb_ast_0)
    (is_asset_not_damaged sct_ast_0)
    (is_asset_not_damaged shp_sal_ast_0)
    ;not moving (is_asset_not_moving)
    (is_asset_not_moving deb_ast_0)
    (is_asset_not_moving sct_ast_0)
    (is_asset_not_moving shp_sal_ast_0)
    ;debris asset predicates
    ;not removing debris (is_debris_asset_not_removing_debris)
    (is_debris_asset_not_removing_debris deb_ast_0)
    ;not unloading debris (is_debris_asset_not_unloading_debris)
    (is_debris_asset_not_unloading_debris deb_ast_0)
    ;scout asset predicates
    ;not scouting (is_scout_asset_not_scouting)
    (is_scout_asset_not_scouting sct_ast_0)
    ;ship salvage asset predicates
    ;not docking (is_ship_salvage_asset_not_docking_ship)
    (is_ship_salvage_asset_not_docking_ship shp_sal_ast_0)
    ;not salvaging (is_ship_salvage_asset_not_salvaging_ship)
    (is_ship_salvage_asset_not_salvaging_ship shp_sal_ast_0)
    ;not towing (is_ship_salvage_asset_not_towing_ship)
    (is_ship_salvage_asset_not_towing_ship shp_sal_ast_0)
    ;location (at)
    (at deb_ast_0 wpt_ini)
    (at sct_ast_0 wpt_ini)
    (at shp_sal_ast_0 wpt_ini)
    ;common functions
    ;speed [km / h] (asset_speed)
    (= (asset_speed deb_ast_0) 10)
    (= (asset_speed sct_ast_0) 20)
    (= (asset_speed shp_sal_ast_0) 10)
    ;debris asset functions
    ;debris capacity [kg] (debris_asset_debris_capacity)
    (= (debris_asset_debris_capacity deb_ast_0) 1000)
    ;debris unload rate [kg / h] (debris_asset_debris_unload_rate)
    (= (debris_asset_debris_unload_rate deb_ast_0) 2000)
    ;debris asset debris weight [kg] (debris_asset_debris_weight)
    (= (debris_asset_debris_weight deb_ast_0) 0)
    ;normal debris removal rate [kg / h] (debris_asset_normal_debris_removal_rate)
    (= (debris_asset_normal_debris_removal_rate deb_ast_0) 100)
    ;underwater debris removal rate [kg / h] (debris_asset_underwater_debris_removal_rate)
    (= (debris_asset_underwater_debris_removal_rate deb_ast_0) 50)
    ;scout asset functions
    ;scout time [h] (scout_asset_scout_time)
    (= (scout_asset_scout_time sct_ast_0) 1)
    ;ship salvage asset functions
    ;dock time [h] (ship_salvage_asset_dock_time)
    (= (ship_salvage_asset_dock_time shp_sal_ast_0) 1)
    ;salvage time [h] (ship_salvage_asset_salvage_time)
    (= (ship_salvage_asset_salvage_time shp_sal_ast_0) 1)

    ;define debris predicates / functions
    ;normal debris location (at)
    (at n_deb_ini_a_0 wpt_ini)
    (at n_deb_ini_a_0 wpt_a_0)
    (at n_deb_a_0_end wpt_a_0)
    (at n_deb_a_0_end wpt_end)
    ;underwater debris location (at)
    (at u_deb_ini_b_0 wpt_ini)
    (at u_deb_ini_b_0 wpt_b_0)
    (at u_deb_b_0_end wpt_b_0)
    (at u_deb_b_0_end wpt_end)
    ;normal debris weight [kg] (debris_weight)
    (= (debris_weight n_deb_ini_a_0) 100)
    (= (debris_weight n_deb_a_0_end) 100)
    ;underwater debris weight [kg] (debris_weight)
    (= (debris_weight u_deb_ini_b_0) 100)
    (= (debris_weight u_deb_b_0_end) 100)

    ;define authority predicates / functions
    ;authority over (has_authority_over_location)
    (has_authority_over_location aut_a wpt_a_0)
    (has_authority_over_location aut_b wpt_b_0)
    ;reply time [h]
    (= (authority_reply_time aut_a) 0.1)
    (= (authority_reply_time aut_b) 0.1)
)

(:goal (and
    ; (at shp_0 shp_dck_0)
    (at deb_ast_0 wpt_end)
    (at u_deb_ini_b_0 wpt_b_0)
))

;un-comment the following line if metric is needed
(:metric minimize (total-time))
)