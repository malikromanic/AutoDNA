$fn = 48;

//osnovne nastavitve
eps = 0.1;

//zunanje mere ohisja
outer_len = 48;
outer_wid = 22;
outer_hgt = 10;
outer_rad = 5;

//debelina sten
wall = 2.2;

//preboji odprtin
cavity_breakthrough = 0.8;

//mere odprtin
charge_port_w = 8;
charge_port_h = 4;
button_hole_d = 4.2;
sensor_open_d = 10;

//izklopljene reze za pas
add_strap_slots = false;

//notranje mere
inner_len = outer_len - 2*wall;
inner_wid = outer_wid - 2*wall;
inner_hgt = outer_hgt - 2*wall;
inner_rad = outer_rad - wall;

//polozaj notranje votline
cavity_z0 = wall;
cavity_z1 = wall + inner_hgt;

//meje zunanjega telesa
x_min = -outer_len/2;
x_max =  outer_len/2;
y_min = -outer_wid/2;
y_max =  outer_wid/2;
z_min = 0;
z_max = outer_hgt;

//meje notranje votline
inner_x_min = -inner_len/2;
inner_x_max =  inner_len/2;
inner_y_min = -inner_wid/2;
inner_y_max =  inner_wid/2;

//preverjanje pravilnih mer
assert(wall >= 2.2, "wall must be at least 2.2 mm");
assert(outer_len > 2*wall, "outer_len must be greater than 2*wall");
assert(outer_wid > 2*wall, "outer_wid must be greater than 2*wall");
assert(outer_hgt > 2*wall, "outer_hgt must be greater than 2*wall");
assert(outer_rad > wall, "outer_rad must be greater than wall");
assert(inner_rad > 0, "inner_rad must be positive");
assert(inner_len > 0, "inner_len must be positive");
assert(inner_wid > 0, "inner_wid must be positive");
assert(inner_hgt > 0, "inner_hgt must be positive");
assert(charge_port_w > 0 && charge_port_h > 0, "charging port dimensions must be positive");
assert(button_hole_d > 0, "button_hole_d must be positive");
assert(sensor_open_d > 0, "sensor_open_d must be positive");
assert(cavity_breakthrough >= 0.8, "cavity_breakthrough must be at least 0.8 mm");

assert(charge_port_w <= outer_wid - 2*wall, "charge_port_w too large for side wall region");
assert(charge_port_h <= outer_hgt - 2*wall, "charge_port_h too large for body height");
assert(button_hole_d <= outer_hgt - 2*wall, "button_hole_d too large for body height");
assert(sensor_open_d <= min(inner_len, inner_wid), "sensor_open_d too large for bottom opening");

//zaobljena skatla
module rounded_box_xy(len, wid, hgt, rad) {
    hull() {
        for (x = [-len/2 + rad, len/2 - rad])
            for (y = [-wid/2 + rad, wid/2 - rad])
                translate([x, y, 0])
                    cylinder(h = hgt, r = rad);
    }
}

//zunanje telo
module outer_shell() {
    rounded_box_xy(outer_len, outer_wid, outer_hgt, outer_rad);
}

//notranja votlina
module inner_cavity() {
    translate([0, 0, cavity_z0])
        rounded_box_xy(inner_len, inner_wid, inner_hgt, inner_rad);
}

//odprtina za polnjenje
module charging_port_cutout() {
    charge_cut_depth = wall + cavity_breakthrough + eps;

    translate([x_max - charge_cut_depth/2 + eps/2, 0, outer_hgt/2])
        cube([charge_cut_depth + eps, charge_port_w, charge_port_h], center = true);
}

//odprtina za gumb
module button_opening_cutout() {
    button_cut_depth = wall + cavity_breakthrough + eps;

    translate([0, y_max - button_cut_depth/2 + eps/2, outer_hgt/2])
        rotate([90, 0, 0])
            cylinder(h = button_cut_depth + eps, d = button_hole_d, center = true);
}

//spodnja odprtina za senzor
module sensor_opening_cutout() {
    sensor_cut_height = wall + cavity_breakthrough + eps;

    translate([0, 0, -eps])
        cylinder(h = sensor_cut_height + eps, d = sensor_open_d);
}

//koncno ohisje
module activity_tracker_casing() {
    difference() {
        outer_shell();

        inner_cavity();

        charging_port_cutout();
        button_opening_cutout();
        sensor_opening_cutout();
    }
}

//prikaz modela
activity_tracker_casing();