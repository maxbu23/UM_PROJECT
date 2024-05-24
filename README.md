Link to data: https://cernbox.cern.ch/files/link/public/jmfnfO2IyI8SNBC?tiles-size=1&items-per-page=100&view-mode=resource-table

Meaning of parameters:

    jmuon_E - particle energy reconstructed using Jmuon algoritm

    jmuon_JENERGY_ENERGY - reconstructed particle energy

    jmuon_JENERGY_CHI2 - chi2 from reconstructed energy fit
    
    jmuon_JENERGY_NDF - number of degrees of freedom from reconstructed energy fit

    jmuon_t - time

    jmuon_likelihood - likelihood of track reconstruction fit

    jmuon_pos_x, jmuon_pos_y, jmuon_pos_z - particle position, coordinates x, y, z

    jmuon_dir_x, jmuon_dir_y, jmuon_dir_z - particle direction, coordinates x, y, z

    jmuon_JSHOWERFIT_ENERGY - shower energy

    jmuon_JGANDALF_CHI2 - chi2 from energy reconstruction fit using JGandalf toolkit

    jmuon_JGANDALF_BETA0_RAD, jmuon_JGANDALF_BETA1_RAD - parameters from fit

    jmuon_JGANDALF_NUMBER_OF_HITS - number of hits from which particle track is reconstructed

    jmuon_AASHOWERFIT_ENERGY - shower energy from AANET toolkit



True (genereted) variables needed to estimate efficiency of regressions:

    energy  - true particle energy

    dir_x, dir_y, dir_z - true particle direction, coordinates x, y, z




Tasks:

************************************************************************

Classification of event neutrino types using KNN and random forests

    There are eight neutrino classes: 
             atm_neutrino_classA.h5  atm_neutrino_classE.h5  
             atm_neutrino_classB.h5  atm_neutrino_classF.h5 
             atm_neutrino_classC.h5  atm_neutrino_classG.h5
             atm_neutrino_classD.h5  atm_neutrino_classH.h5

    Variables to use: jmuon_E, jmuon_t, jmuon_likelihood, jmuon_JGANDALF_CHI2,
                      jmuon_pos_x, jmuon_pos_y, jmuon_pos_z,
                      jmuon_dir_x, jmuon_dir_y, jmuon_dir_z

    To read files:  import pandas as pd
                    df = pd.read_hdf("file.h5", "y")

************************************************************************

Classification atmosferic muon vs neutrino using GBDT and random forests

    File for muon class: atm_muon.h5

    We want to disantangle between atmosferic muons and nutrinos (does not matter what kind of neutrino class is).
    To create one neutrino class concatenate all neutrino types:
             atm_neutrino_classA.h5  atm_neutrino_classE.h5
             atm_neutrino_classB.h5  atm_neutrino_classF.h5
             atm_neutrino_classC.h5  atm_neutrino_classG.h5  
             atm_neutrino_classD.h5  atm_neutrino_classH.h5

    Variables to use:  jmuon_E, jmuon_t, jmuon_likelihood,
                       jmuon_pos_x, jmuon_pos_y, jmuon_pos_z,
                       jmuon_dir_x, jmuon_dir_y, jmuon_dir_z,
                       jmuon_JGANDALF_BETA0_RAD, jmuon_JGANDALF_BETA1_RAD, jmuon_JGANDALF_CHI2,
                       jmuon_JSHOWERFIT_ENERGY

    To read files:  import pandas as pd
                    df = pd.read_hdf("file.h5", "y")

************************************************************************

Neutrino energy regression using DNN

    Treat each neutrino class separately. This way allows to check energy differences
    between neutrino classes if they are.

    There are eight neutrino classes:
             atm_neutrino_classA.h5  atm_neutrino_classE.h5
             atm_neutrino_classB.h5  atm_neutrino_classF.h5
             atm_neutrino_classC.h5  atm_neutrino_classG.h5
             atm_neutrino_classD.h5  atm_neutrino_classH.h5
 
    There are different algorythms for energy reconstruction. Since the efficiency of each algorythm
    may differ the model should be trained with values obtained from each technique:
 
            reconstructed_energy = f(jmuon_E, jmuon_JENERGY_ENERGY, jmuon_JENERGY_CHI2, jmuon_JENERGY_NDF,
                                     jmuon_JGANDALF_NUMBER_OF_HITS, jmuon_JSHOWERFIT_ENERGY,
                                     jmuon_AASHOWERFIT_ENERGY)

    Predicted energy can be compared with true (used in event generation) energy:  energy

    To read files:  import pandas as pd
                    df = pd.read_hdf("file.h5", "y")

************************************************************************
                    
Neutrino direction regression using GBDT

    Treat each neutrino class separately. It allows to check whether different neutrinos
    have the same or different direction.

    There are eight neutrino classes:
             atm_neutrino_classA.h5  atm_neutrino_classE.h5
             atm_neutrino_classB.h5  atm_neutrino_classF.h5
             atm_neutrino_classC.h5  atm_neutrino_classG.h5
             atm_neutrino_classD.h5  atm_neutrino_classH.h5
 
    Train the model with parameters:
            (reco_dir_x, reco_dir_y, reco_dir_z) = f(jmuon_dir_x, jmuon_dir_y, jmuon_dir_z, jmuon_likelihood)

    Predicted direction can be compared with true direction:  true_direction = f(dir_x, dir_y, dir_z)

    To read files:  import pandas as pd
                    df = pd.read_hdf("file.h5", "y")

************************************************************************

Simultaneous regression of energy and direction of neutrino types using CNN (group with three studends)

    Treat each neutrino class separately. It allows to check whether different neutrinos
    have the same or different energy and direction.
    
    There are eight neutrino classes:
             atm_neutrino_classA.h5  atm_neutrino_classE.h5
             atm_neutrino_classB.h5  atm_neutrino_classF.h5
             atm_neutrino_classC.h5  atm_neutrino_classG.h5
             atm_neutrino_classD.h5  atm_neutrino_classH.h5
                                                                                   
    Train model simultaneously with reconstruceted energy and reconstructed direction:
                   reco_energy = f(jmuon_E, jmuon_JENERGY_ENERGY, jmuon_JENERGY_CHI2, jmuon_JENERGY_NDF
                                   jmuon_JGANDALF_NUMBER_OF_HITS, jmuon_JSHOWERFIT_ENERGY,
                                   jmuon_AASHOWERFIT_ENERGY)
             (reco_dir_x, reco_dir_y, reco_dir_z) = f(jmuon_dir_x, jmuon_dir_y, jmuon_dir_z, jmuon_likelihood)

    Predicted energy and direction can be compared with corresponding true values:
                      true_energy = f(energy)
                      true_direction = f(dir_x, dir_y, dir_z)

    To read file:  import pandas as pd
                   df = pd.read_hdf("file.h5", "y")

************************************************************************

Simultaneous regression of energy and direction of atmosferic muon (group with three studends) using GBDT

    File to use: atm_muon.h5

    Train model simultaneously with reconstruceted energy and reconstructed direction:
                   reco_energy = f(jmuon_E, jmuon_JENERGY_ENERGY, jmuon_JENERGY_CHI2, jmuon_JENERGY_NDF,
                                   jmuon_JGANDALF_NUMBER_OF_HITS, jmuon_JSHOWERFIT_ENERGY,
                                   jmuon_AASHOWERFIT_ENERGY)
             (reco_dir_x, reco_dir_y, reco_dir_z) = f(jmuon_dir_x, jmuon_dir_y, jmuon_dir_z, jmuon_likelihood)

    Predicted energy and direction can be compared with corresponding true values:
                      true_energy = f(energy)
                      true_direction = f(dir_x, dir_y, dir_z)

    To read file:  import pandas as pd
                   df = pd.read_hdf("file.h5", "y")

************************************************************************

Atmosferic muon direction regression using CNN

    File to use: atm_muon.h5

    Train model with reconstructed direction parameters: 
            (reco_dir_x, reco_dir_y, reco_dir_z) = f(jmuon_dir_x, jmuon_dir_y, jmuon_dir_z, jmuon_likelihood)

    Predicted direction can be compared with true direction: true_direction = f(dir_x, dir_y, dir_z)

    To read file:  import pandas as pd
                   df = pd.read_hdf("file.h5", "y")

************************************************************************
  
# UM_PROJECT
# UM_PROJECT
