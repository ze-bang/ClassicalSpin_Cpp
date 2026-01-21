/**
 * unitcell_builders.h - Unit cell builder function declarations
 * 
 * This header declares the builder functions for various lattice types.
 * Implementations are in unitcell_builders.cpp.
 */

#ifndef UNITCELL_BUILDERS_H
#define UNITCELL_BUILDERS_H

#include "spin_config.h"
#include "unitcell.h"

// Build BCAO honeycomb unit cell
UnitCell build_bcao_honeycomb(const SpinConfig& config);

// Build Kitaev honeycomb unit cell
UnitCell build_kitaev_honeycomb(const SpinConfig& config);

// Build pyrochlore unit cell
UnitCell build_pyrochlore(const SpinConfig& config);

// Build non-Kramers pyrochlore unit cell (Jpm, Jzz, Jpmpm exchange)
UnitCell build_pyrochlore_non_kramer(const SpinConfig& config);

// Build TmFeO3 mixed unit cell (SU2 Fe + SU3 Tm)
MixedUnitCell build_tmfeo3(const SpinConfig& config);

// Build TmFeO3 Fe-only unit cell (SU2 only, no Tm atoms)
UnitCell build_tmfeo3_fe(const SpinConfig& config);

// Build TmFeO3 Tm-only unit cell (SU3 only, no Fe atoms)
UnitCell build_tmfeo3_tm(const SpinConfig& config);

#endif // UNITCELL_BUILDERS_H
