# -----------------------------------------------------
# Quentin HUAN 06/2024
# 
# This script generates all the panels used in mitsuba scenes.
# (See generator.py for details)
# -----------------------------------------------------
from generator import *

outfolder="./panels/"

def various():
    ## cloud_5_e0.06
    generate_panel(outfolder, "cloud_5_e0.06", seed=5, e=0.06, heightmap_front="heightmaps/cloud_blur_2x2.exr",heightmap_back="heightmaps/cloud_blur_2x2.exr", H1=0.0005, H2=0.0005, bubble_shape=(50,50,1), radius_range=(0.001, 0.002), bubble_density=0.25, RIF_shape=(1024,1024,2), chord_throughtput= 0.004, chord_protuberance=0.0005, eta_base=1.54, delta_eta=0.0005, gathering_zone=0.3, restore_RIF_from_file=False, debug=False)

    ## manchon_2_e0.002
    generate_panel(outfolder, "manchon_2_e0.002", seed=2, e=0.002, heightmap_front="heightmaps/manchon_1_1024.exr",heightmap_back="heightmaps/manchon_2_1024.exr", H1=0.0005, H2=0.0005, bubble_shape=(50,50,1), radius_range=(0.0001, 0.001), bubble_density=0.25, RIF_shape=(1024,1024,2), chord_throughtput= 0.004, chord_protuberance=0.0005, eta_base=1.54, delta_eta=0.0005, gathering_zone=0.3, restore_RIF_from_file=False, debug=False)

    ## manchon_2_e0.0025
    generate_panel(outfolder, "manchon_2_e0.0025", seed=2, e=0.0025, heightmap_front="heightmaps/manchon_1_1024.exr",heightmap_back="heightmaps/manchon_2_1024.exr", H1=0.0005, H2=0.0005, bubble_shape=(50,50,1), radius_range=(0.0001, 0.001), bubble_density=0.25, RIF_shape=(1024,1024,2), chord_throughtput= 0.004, chord_protuberance=0.0005, eta_base=1.54, delta_eta=0.0005, gathering_zone=0.3, restore_RIF_from_file=False, debug=False)


#######################################################################
#                      Notre Dame bridge panels                       #
#######################################################################

def PontNotreDame():
    ## manchon_2_e0.01
    generate_panel(outfolder, "manchon_2_e0.01", seed=2, e=0.01, heightmap_front="heightmaps/manchon_1_1024.exr",heightmap_back="heightmaps/manchon_2_1024.exr", H1=0.0005, H2=0.0005, bubble_shape=(50,50,1), radius_range=(0.001, 0.002), bubble_density=0.25, RIF_shape=(1024,1024,2), chord_throughtput= 0.004, chord_protuberance=0.0005, eta_base=1.54, delta_eta=0.0005, gathering_zone=0.3, restore_RIF_from_file=False, debug=False)

    ## turbulence_2_e0.01
    generate_panel(outfolder, "turbulence_2_e0.01", seed=2, e=0.01, heightmap_front="heightmaps/Turbulence.exr",heightmap_back="heightmaps/Turbulence.exr", H1=0.001, H2=0.001, bubble_shape=(50,50,1), radius_range=(0.001, 0.002), bubble_density=0.25, RIF_shape=(1024,1024,2), chord_throughtput= 0.004, chord_protuberance=0.0005, eta_base=1.54, delta_eta=0.05, gathering_zone=0.5, restore_RIF_from_file=False, debug=False)

    ## manchon_2_e0.02
    generate_panel(outfolder, "manchon_2_e0.02", seed=2, e=0.02, heightmap_front="heightmaps/manchon_1_1024.exr",heightmap_back="heightmaps/manchon_2_1024.exr", H1=0.0005, H2=0.0005, bubble_shape=(50,50,1), radius_range=(0.001, 0.002), bubble_density=0.25, RIF_shape=(1024,1024,2), chord_throughtput= 0.004, chord_protuberance=0.0005, eta_base=1.54, delta_eta=0.0005, gathering_zone=0.3, restore_RIF_from_file=False, debug=False)

def stained_glass():
    ## cloud_5_e0.04
    generate_panel(outfolder, "cloud_5_e0.04", seed=5, e=0.04, heightmap_front="heightmaps/cloud_blur_2x2.exr",heightmap_back="heightmaps/cloud_blur_2x2.exr", H1=0.0005, H2=0.0005, bubble_shape=(50,50,1), radius_range=(0.0001, 0.0025), bubble_density=0.5, RIF_shape=(1024,1024,4), chord_throughtput= 0.004, chord_protuberance=0.0005, eta_base=1.54, delta_eta=0.0005, gathering_zone=0.3, restore_RIF_from_file=False, debug=False)
    generate_panel(outfolder, "cloud_5_e0.02", seed=5, e=0.02, heightmap_front="heightmaps/cloud_blur_2x2.exr",heightmap_back="heightmaps/cloud_blur_2x2.exr", H1=0.0005, H2=0.0005, bubble_shape=(50,50,1), radius_range=(0.0001, 0.0025), bubble_density=0.5, RIF_shape=(1024,1024,2), chord_throughtput= 0.004, chord_protuberance=0.0005, eta_base=1.54, delta_eta=0.0005, gathering_zone=0.3, restore_RIF_from_file=False, debug=False)
    generate_panel(outfolder, "crown_5_e0.02", seed=5, e=0.04, heightmap_front="heightmaps/crown.exr",heightmap_back="heightmaps/crown.exr", H1=0.0005, H2=0.0005, bubble_shape=(50,50,1), radius_range=(0.0001, 0.0025), bubble_density=0.1, RIF_shape=(1024,1024,2), chord_throughtput= 0.004, chord_protuberance=0.0000, eta_base=1.54, delta_eta=0.0005, gathering_zone=0.3, restore_RIF_from_file=False, debug=False)


#######################################################################
#           Caustic vitrail e0.03 full and isolated effects           #
#######################################################################
def caustic_vitrail():
    heightmap_front="heightmaps/manchon_1_1024.exr"
    heightmap_back="heightmaps/manchon_2_1024.exr"

    H1 = 0.001
    H2 = 0.001
    RIF_shape=(1024,1024,2)
    bubble_shape = (64,64,1)
    chord_throughtput= 0.0008
    chord_protuberance=0.0002


    ### manchon_2_e0.03_surfaceOnly
    #  generate_panel(outfolder, "manchon_2_e0.03_surfaceOnly", seed=2, e=0.03, heightmap_front=heightmap_front,heightmap_back=heightmap_back, H1=H1, H2=H2, bubble_shape=bubble_shape, radius_range=(0.001, 0.002), bubble_density=0.25, RIF_shape=RIF_shape, chord_throughtput=chord_throughtput, chord_protuberance=chord_protuberance, eta_base=1.54, delta_eta=0.0005, gathering_zone=0.3, restore_RIF_from_file=False, debug=False, no_bubbles=True, no_chords=True, no_surface=False)

    ### manchon_2_e0.03_surface_bubbles
    #  generate_panel(outfolder, "manchon_2_e0.03_surface_bubbles", seed=2, e=0.03, heightmap_front=heightmap_front,heightmap_back=heightmap_back, H1=H1, H2=H2, bubble_shape=bubble_shape, radius_range=(0.001, 0.002), bubble_density=0.25, RIF_shape=RIF_shape, chord_throughtput= chord_throughtput, chord_protuberance=chord_protuberance, eta_base=1.54, delta_eta=0.0005, gathering_zone=0.3, restore_RIF_from_file=False, debug=False, no_bubbles=False, no_chords=True, no_surface=False)

    ### manchon_2_e0.03_full
    #  generate_panel(outfolder, "manchon_2_e0.03_full", seed=2, e=0.03, heightmap_front=heightmap_front,heightmap_back=heightmap_back, H1=H1, H2=H2, bubble_shape=bubble_shape, radius_range=(0.001, 0.002), bubble_density=0.25, RIF_shape=RIF_shape, chord_throughtput=chord_throughtput, chord_protuberance=chord_protuberance, eta_base=1.54, delta_eta=0.0005, gathering_zone=0.3, restore_RIF_from_file=False, debug=False)


    ### exagerated
    H1 = 0.001 
    H2 = 0.001 
    e=0.03
    RIF_shape=(1024,1024,2)
    print(RIF_shape)
    bubble_shape = (64,64,1)
    chord_throughtput= 0.001
    chord_protuberance=0.0001
    seed=2

    ## manchon_2_e0.03_surfaceOnly
    #  generate_panel(outfolder, "manchon_2_exagerated_e0.03_surfaceOnly", seed=seed, e=0.03, heightmap_front=heightmap_front,heightmap_back=heightmap_back, H1=H1, H2=H2, bubble_shape=bubble_shape, radius_range=(0.002, 0.005), bubble_density=0.85, RIF_shape=RIF_shape, chord_throughtput=chord_throughtput, chord_protuberance=chord_protuberance, eta_base=1.54, delta_eta=0.05, gathering_zone=0.3, restore_RIF_from_file=False, debug=False, no_bubbles=True, no_chords=True, no_surface=False)
    ## manchon_2_e0.03_surface_bubbles
    #  generate_panel(outfolder, "manchon_2_exagerated_e0.03_surface_bubbles", seed=seed, e=0.03, heightmap_front=heightmap_front,heightmap_back=heightmap_back, H1=H1, H2=H2, bubble_shape=bubble_shape, radius_range=(0.002, 0.005), bubble_density=0.85, RIF_shape=RIF_shape, chord_throughtput= chord_throughtput, chord_protuberance=chord_protuberance, eta_base=1.54, delta_eta=0.05, gathering_zone=0.3, restore_RIF_from_file=False, debug=False, no_bubbles=False, no_chords=True, no_surface=False)

    ## manchon_2_e0.03_full
    generate_panel(outfolder, "manchon_2_exagerated_e0.03_full", seed=seed, e=0.03, heightmap_front=heightmap_front,heightmap_back=heightmap_back, H1=H1, H2=H2, bubble_shape=bubble_shape, radius_range=(0.002, 0.005), bubble_density=0.85, RIF_shape=RIF_shape, chord_throughtput=chord_throughtput, chord_protuberance=chord_protuberance, eta_base=1.54, delta_eta=0.1, gathering_zone=0.3, restore_RIF_from_file=False, debug=False)

    ## manchon_2_e0.03_full_lowres
    RIF_shape=(256,256,2)
    generate_panel(outfolder, "manchon_2_exagerated_e0.03_full_lowres", seed=seed, e=0.03, heightmap_front=heightmap_front,heightmap_back=heightmap_back, H1=H1, H2=H2, bubble_shape=bubble_shape, radius_range=(0.002, 0.005), bubble_density=0.85, RIF_shape=RIF_shape, chord_throughtput=chord_throughtput, chord_protuberance=chord_protuberance, eta_base=1.54, delta_eta=0.1, gathering_zone=0.3, restore_RIF_from_file=False, debug=False)


#  various()
#  PontNotreDame()
#  stained_glass()
caustic_vitrail()






