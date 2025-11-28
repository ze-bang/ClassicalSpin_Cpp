#ifndef HDF5_IO_H
#define HDF5_IO_H

#include <H5Cpp.h>
#include <vector>
#include <string>
#include <Eigen/Dense>
#include <ctime>
#include <sstream>
#include <iomanip>
#include "simple_linear_alg.h"

/**
 * HDF5 Writer for molecular dynamics trajectories
 * 
 * This class provides efficient I/O for MD simulations by storing
 * all data in a single HDF5 file with comprehensive metadata.
 * 
 * File structure:
 * /trajectory/
 *   - times [n_steps]                         : Time points
 *   - spins [n_steps, n_sites, spin_dim]      : Full spin configuration
 *   - magnetization_antiferro [n_steps, spin_dim] : Antiferromagnetic order parameter
 *   - magnetization_local [n_steps, spin_dim]     : Local magnetization
 * 
 * /metadata/
 *   - lattice_size          : Total number of sites
 *   - spin_dim              : Dimension of spin vectors
 *   - n_atoms               : Number of atoms per unit cell
 *   - dimensions [3]        : Lattice dimensions (dim1, dim2, dim3)
 *   - integration_method    : ODE solver used (e.g., "dopri5", "rk4")
 *   - dt_initial            : Initial time step
 *   - T_start               : Integration start time
 *   - T_end                 : Integration end time
 *   - save_interval         : Steps between saves
 *   - spin_length           : Magnitude of spin vectors
 *   - creation_time         : ISO 8601 timestamp
 *   - git_commit            : Git commit hash (if available)
 *   - code_version          : Version string
 */
class HDF5MDWriter {
public:
    HDF5MDWriter(const std::string& filename, 
                 size_t lattice_size, 
                 size_t spin_dim,
                 size_t n_atoms,
                 size_t dim1, size_t dim2, size_t dim3,
                 const std::string& method,
                 double dt_initial,
                 double T_start,
                 double T_end,
                 size_t save_interval,
                 float spin_length = 1.0,
                 const std::vector<Eigen::Vector3d>* positions = nullptr,
                 size_t reserve_steps = 1000)
        : filename_(filename),
          lattice_size_(lattice_size),
          spin_dim_(spin_dim),
          current_step_(0)
    {
        // Create HDF5 file
        file_ = H5::H5File(filename, H5F_ACC_TRUNC);
        
        // Create groups
        trajectory_group_ = file_.createGroup("/trajectory");
        metadata_group_ = file_.createGroup("/metadata");
        
        // Get current timestamp
        std::time_t now = std::time(nullptr);
        char time_str[100];
        std::strftime(time_str, sizeof(time_str), "%Y-%m-%dT%H:%M:%S", std::localtime(&now));
        
        // Write comprehensive metadata
        write_scalar_attribute(metadata_group_, "lattice_size", lattice_size);
        write_scalar_attribute(metadata_group_, "spin_dim", spin_dim);
        write_scalar_attribute(metadata_group_, "n_atoms", n_atoms);
        write_string_attribute(metadata_group_, "integration_method", method);
        write_double_attribute(metadata_group_, "dt_initial", dt_initial);
        write_double_attribute(metadata_group_, "T_start", T_start);
        write_double_attribute(metadata_group_, "T_end", T_end);
        write_scalar_attribute(metadata_group_, "save_interval", save_interval);
        write_double_attribute(metadata_group_, "spin_length", spin_length);
        write_string_attribute(metadata_group_, "creation_time", std::string(time_str));
        write_string_attribute(metadata_group_, "code_version", "ClassicalSpin_Cpp v1.0");
        write_string_attribute(metadata_group_, "file_format", "HDF5_MD_v1.0");
        
        // Write lattice dimensions array
        hsize_t dims_data[1] = {3};
        H5::DataSpace dims_space(1, dims_data);
        H5::DataSet dims_dataset = metadata_group_.createDataSet(
            "dimensions", H5::PredType::NATIVE_HSIZE, dims_space);
        hsize_t dimensions[3] = {dim1, dim2, dim3};
        dims_dataset.write(dimensions, H5::PredType::NATIVE_HSIZE);
        
        // Write site positions if provided [lattice_size, 3]
        if (positions != nullptr && positions->size() == lattice_size) {
            hsize_t pos_dims[2] = {lattice_size, 3};
            H5::DataSpace pos_space(2, pos_dims);
            H5::DataSet pos_dataset = metadata_group_.createDataSet(
                "positions", H5::PredType::NATIVE_DOUBLE, pos_space);
            
            std::vector<double> pos_data(lattice_size * 3);
            for (size_t i = 0; i < lattice_size; ++i) {
                pos_data[i * 3 + 0] = (*positions)[i](0);
                pos_data[i * 3 + 1] = (*positions)[i](1);
                pos_data[i * 3 + 2] = (*positions)[i](2);
            }
            pos_dataset.write(pos_data.data(), H5::PredType::NATIVE_DOUBLE);
        }
        
        // Reserve space for trajectory data (expandable datasets)
        // Times dataset
        hsize_t time_dims[1] = {0};
        hsize_t time_maxdims[1] = {H5S_UNLIMITED};
        hsize_t time_chunk[1] = {1000};
        H5::DataSpace time_space(1, time_dims, time_maxdims);
        H5::DSetCreatPropList time_prop;
        time_prop.setChunk(1, time_chunk);
        time_prop.setDeflate(6); // Compression level
        times_dataset_ = trajectory_group_.createDataSet(
            "times", H5::PredType::NATIVE_DOUBLE, time_space, time_prop);
        
        // Magnetization antiferro dataset [n_steps, spin_dim]
        hsize_t mag_dims[2] = {0, spin_dim};
        hsize_t mag_maxdims[2] = {H5S_UNLIMITED, spin_dim};
        hsize_t mag_chunk[2] = {100, spin_dim};
        H5::DataSpace mag_space(2, mag_dims, mag_maxdims);
        H5::DSetCreatPropList mag_prop;
        mag_prop.setChunk(2, mag_chunk);
        mag_prop.setDeflate(6);
        mag_antiferro_dataset_ = trajectory_group_.createDataSet(
            "magnetization_antiferro", H5::PredType::NATIVE_DOUBLE, mag_space, mag_prop);
        
        // Magnetization local dataset [n_steps, spin_dim]
        mag_local_dataset_ = trajectory_group_.createDataSet(
            "magnetization_local", H5::PredType::NATIVE_DOUBLE, mag_space, mag_prop);
        
        // Spins dataset [n_steps, n_sites, spin_dim] - optional, can be huge
        // Only create if user wants full trajectory
        hsize_t spin_dims[3] = {0, lattice_size, spin_dim};
        hsize_t spin_maxdims[3] = {H5S_UNLIMITED, lattice_size, spin_dim};
        hsize_t spin_chunk[3] = {1, lattice_size, spin_dim};
        H5::DataSpace spin_space(3, spin_dims, spin_maxdims);
        H5::DSetCreatPropList spin_prop;
        spin_prop.setChunk(3, spin_chunk);
        spin_prop.setDeflate(6);
        spins_dataset_ = trajectory_group_.createDataSet(
            "spins", H5::PredType::NATIVE_DOUBLE, spin_space, spin_prop);
    }
    
    /**
     * Write a single time step
     */
    void write_step(double time, 
                   const SpinVector& mag_antiferro,
                   const SpinVector& mag_local,
                   const std::vector<SpinVector>& spins) {
        // Extend datasets
        hsize_t new_size[3];
        
        // Write time
        new_size[0] = current_step_ + 1;
        times_dataset_.extend(new_size);
        H5::DataSpace time_fspace = times_dataset_.getSpace();
        hsize_t time_offset[1] = {current_step_};
        hsize_t time_count[1] = {1};
        time_fspace.selectHyperslab(H5S_SELECT_SET, time_count, time_offset);
        H5::DataSpace time_mspace(1, time_count);
        times_dataset_.write(&time, H5::PredType::NATIVE_DOUBLE, time_mspace, time_fspace);
        
        // Write magnetization antiferro
        new_size[0] = current_step_ + 1;
        new_size[1] = spin_dim_;
        mag_antiferro_dataset_.extend(new_size);
        H5::DataSpace mag_af_fspace = mag_antiferro_dataset_.getSpace();
        hsize_t mag_offset[2] = {current_step_, 0};
        hsize_t mag_count[2] = {1, spin_dim_};
        mag_af_fspace.selectHyperslab(H5S_SELECT_SET, mag_count, mag_offset);
        H5::DataSpace mag_mspace(2, mag_count);
        std::vector<double> mag_af_data(spin_dim_);
        for (size_t i = 0; i < spin_dim_; ++i) {
            mag_af_data[i] = mag_antiferro(i);
        }
        mag_antiferro_dataset_.write(mag_af_data.data(), H5::PredType::NATIVE_DOUBLE, 
                                     mag_mspace, mag_af_fspace);
        
        // Write magnetization local
        mag_local_dataset_.extend(new_size);
        H5::DataSpace mag_local_fspace = mag_local_dataset_.getSpace();
        mag_local_fspace.selectHyperslab(H5S_SELECT_SET, mag_count, mag_offset);
        std::vector<double> mag_local_data(spin_dim_);
        for (size_t i = 0; i < spin_dim_; ++i) {
            mag_local_data[i] = mag_local(i);
        }
        mag_local_dataset_.write(mag_local_data.data(), H5::PredType::NATIVE_DOUBLE, 
                                mag_mspace, mag_local_fspace);
        
        // Write spins
        new_size[0] = current_step_ + 1;
        new_size[1] = lattice_size_;
        new_size[2] = spin_dim_;
        spins_dataset_.extend(new_size);
        H5::DataSpace spin_fspace = spins_dataset_.getSpace();
        hsize_t spin_offset[3] = {current_step_, 0, 0};
        hsize_t spin_count[3] = {1, lattice_size_, spin_dim_};
        spin_fspace.selectHyperslab(H5S_SELECT_SET, spin_count, spin_offset);
        H5::DataSpace spin_mspace(3, spin_count);
        
        // Flatten spins data
        std::vector<double> spin_data(lattice_size_ * spin_dim_);
        for (size_t i = 0; i < lattice_size_; ++i) {
            for (size_t j = 0; j < spin_dim_; ++j) {
                spin_data[i * spin_dim_ + j] = spins[i](j);
            }
        }
        spins_dataset_.write(spin_data.data(), H5::PredType::NATIVE_DOUBLE, 
                            spin_mspace, spin_fspace);
        
        current_step_++;
    }

    /**
     * Write a single time step directly from flat state array (zero-copy optimization)
     * This method avoids the SpinVector conversion entirely
     */
    void write_flat_step(double time, 
                        const SpinVector& mag_antiferro,
                        const SpinVector& mag_local,
                        const double* flat_spins) {
        // Extend datasets
        hsize_t new_size[3];
        
        // Write time
        new_size[0] = current_step_ + 1;
        times_dataset_.extend(new_size);
        H5::DataSpace time_fspace = times_dataset_.getSpace();
        hsize_t time_offset[1] = {current_step_};
        hsize_t time_count[1] = {1};
        time_fspace.selectHyperslab(H5S_SELECT_SET, time_count, time_offset);
        H5::DataSpace time_mspace(1, time_count);
        times_dataset_.write(&time, H5::PredType::NATIVE_DOUBLE, time_mspace, time_fspace);
        
        // Write magnetization antiferro
        new_size[0] = current_step_ + 1;
        new_size[1] = spin_dim_;
        mag_antiferro_dataset_.extend(new_size);
        H5::DataSpace mag_af_fspace = mag_antiferro_dataset_.getSpace();
        hsize_t mag_offset[2] = {current_step_, 0};
        hsize_t mag_count[2] = {1, spin_dim_};
        mag_af_fspace.selectHyperslab(H5S_SELECT_SET, mag_count, mag_offset);
        H5::DataSpace mag_mspace(2, mag_count);
        std::vector<double> mag_af_data(spin_dim_);
        for (size_t i = 0; i < spin_dim_; ++i) {
            mag_af_data[i] = mag_antiferro(i);
        }
        mag_antiferro_dataset_.write(mag_af_data.data(), H5::PredType::NATIVE_DOUBLE, 
                                     mag_mspace, mag_af_fspace);
        
        // Write magnetization local
        mag_local_dataset_.extend(new_size);
        H5::DataSpace mag_local_fspace = mag_local_dataset_.getSpace();
        mag_local_fspace.selectHyperslab(H5S_SELECT_SET, mag_count, mag_offset);
        std::vector<double> mag_local_data(spin_dim_);
        for (size_t i = 0; i < spin_dim_; ++i) {
            mag_local_data[i] = mag_local(i);
        }
        mag_local_dataset_.write(mag_local_data.data(), H5::PredType::NATIVE_DOUBLE, 
                                mag_mspace, mag_local_fspace);
        
        // Write spins directly from flat array (zero-copy)
        new_size[0] = current_step_ + 1;
        new_size[1] = lattice_size_;
        new_size[2] = spin_dim_;
        spins_dataset_.extend(new_size);
        H5::DataSpace spin_fspace = spins_dataset_.getSpace();
        hsize_t spin_offset[3] = {current_step_, 0, 0};
        hsize_t spin_count[3] = {1, lattice_size_, spin_dim_};
        spin_fspace.selectHyperslab(H5S_SELECT_SET, spin_count, spin_offset);
        H5::DataSpace spin_mspace(3, spin_count);
        
        // Write directly from flat array - no copy needed!
        spins_dataset_.write(flat_spins, H5::PredType::NATIVE_DOUBLE, 
                            spin_mspace, spin_fspace);
        
        current_step_++;
    }
    
    /**
     * Close the file and flush all buffers
     */
    void close() {
        times_dataset_.close();
        mag_antiferro_dataset_.close();
        mag_local_dataset_.close();
        spins_dataset_.close();
        trajectory_group_.close();
        metadata_group_.close();
        file_.close();
    }
    
    ~HDF5MDWriter() {
        try {
            if (file_.getId() > 0) {
                close();
            }
        } catch (...) {
            // Ignore errors in destructor
        }
    }
    
private:
    void write_scalar_attribute(H5::Group& group, const std::string& name, hsize_t value) {
        H5::DataSpace attr_space(H5S_SCALAR);
        H5::Attribute attr = group.createAttribute(
            name, H5::PredType::NATIVE_HSIZE, attr_space);
        attr.write(H5::PredType::NATIVE_HSIZE, &value);
    }
    
    void write_double_attribute(H5::Group& group, const std::string& name, double value) {
        H5::DataSpace attr_space(H5S_SCALAR);
        H5::Attribute attr = group.createAttribute(
            name, H5::PredType::NATIVE_DOUBLE, attr_space);
        attr.write(H5::PredType::NATIVE_DOUBLE, &value);
    }
    
    void write_string_attribute(H5::Group& group, const std::string& name, const std::string& value) {
        H5::StrType str_type(H5::PredType::C_S1, value.size() + 1);
        H5::DataSpace attr_space(H5S_SCALAR);
        H5::Attribute attr = group.createAttribute(name, str_type, attr_space);
        attr.write(str_type, value.c_str());
    }
    
    std::string filename_;
    size_t lattice_size_;
    size_t spin_dim_;
    size_t current_step_;
    
    H5::H5File file_;
    H5::Group trajectory_group_;
    H5::Group metadata_group_;
    H5::DataSet times_dataset_;
    H5::DataSet mag_antiferro_dataset_;
    H5::DataSet mag_local_dataset_;
    H5::DataSet spins_dataset_;
};

/**
 * HDF5 Writer for mixed SU(2)/SU(3) lattice molecular dynamics trajectories
 * 
 * File structure:
 * /trajectory_SU2/
 *   - times [n_steps]
 *   - spins [n_steps, n_sites_SU2, spin_dim_SU2]
 *   - magnetization_antiferro [n_steps, spin_dim_SU2]
 *   - magnetization_local [n_steps, spin_dim_SU2]
 * 
 * /trajectory_SU3/
 *   - times [n_steps]
 *   - spins [n_steps, n_sites_SU3, spin_dim_SU3]
 *   - magnetization_antiferro [n_steps, spin_dim_SU3]
 *   - magnetization_local [n_steps, spin_dim_SU3]
 * 
 * /metadata_SU2/ and /metadata_SU3/
 *   - lattice_size, spin_dim, n_atoms, dimensions, spin_length, positions
 * 
 * /metadata_global/
 *   - integration_method, dt_initial, T_start, T_end, save_interval, creation_time, etc.
 */
class HDF5MixedMDWriter {
public:
    HDF5MixedMDWriter(const std::string& filename, 
                      size_t lattice_size_SU2, size_t spin_dim_SU2, size_t n_atoms_SU2,
                      size_t lattice_size_SU3, size_t spin_dim_SU3, size_t n_atoms_SU3,
                      size_t dim1, size_t dim2, size_t dim3,
                      const std::string& method,
                      double dt_initial, double T_start, double T_end,
                      size_t save_interval,
                      float spin_length_SU2 = 1.0,
                      float spin_length_SU3 = 1.0,
                      const std::vector<Eigen::Vector3d>* positions_SU2 = nullptr,
                      const std::vector<Eigen::Vector3d>* positions_SU3 = nullptr,
                      size_t reserve_steps = 1000)
        : filename_(filename),
          lattice_size_SU2_(lattice_size_SU2), spin_dim_SU2_(spin_dim_SU2),
          lattice_size_SU3_(lattice_size_SU3), spin_dim_SU3_(spin_dim_SU3),
          current_step_(0)
    {
        // Create HDF5 file
        file_ = H5::H5File(filename, H5F_ACC_TRUNC);
        
        // Create groups
        traj_SU2_group_ = file_.createGroup("/trajectory_SU2");
        traj_SU3_group_ = file_.createGroup("/trajectory_SU3");
        meta_SU2_group_ = file_.createGroup("/metadata_SU2");
        meta_SU3_group_ = file_.createGroup("/metadata_SU3");
        meta_global_group_ = file_.createGroup("/metadata_global");
        
        // Get current timestamp
        std::time_t now = std::time(nullptr);
        char time_str[100];
        std::strftime(time_str, sizeof(time_str), "%Y-%m-%dT%H:%M:%S", std::localtime(&now));
        
        // Write global metadata
        write_string_attribute(meta_global_group_, "integration_method", method);
        write_double_attribute(meta_global_group_, "dt_initial", dt_initial);
        write_double_attribute(meta_global_group_, "T_start", T_start);
        write_double_attribute(meta_global_group_, "T_end", T_end);
        write_scalar_attribute(meta_global_group_, "save_interval", save_interval);
        write_string_attribute(meta_global_group_, "creation_time", std::string(time_str));
        write_string_attribute(meta_global_group_, "code_version", "ClassicalSpin_Cpp v1.0");
        write_string_attribute(meta_global_group_, "file_format", "HDF5_Mixed_MD_v1.0");
        
        // Write lattice dimensions (shared)
        hsize_t dims_data[1] = {3};
        H5::DataSpace dims_space(1, dims_data);
        H5::DataSet dims_dataset = meta_global_group_.createDataSet(
            "dimensions", H5::PredType::NATIVE_HSIZE, dims_space);
        hsize_t dimensions[3] = {dim1, dim2, dim3};
        dims_dataset.write(dimensions, H5::PredType::NATIVE_HSIZE);
        
        // SU2 metadata
        write_scalar_attribute(meta_SU2_group_, "lattice_size", lattice_size_SU2);
        write_scalar_attribute(meta_SU2_group_, "spin_dim", spin_dim_SU2);
        write_scalar_attribute(meta_SU2_group_, "n_atoms", n_atoms_SU2);
        write_double_attribute(meta_SU2_group_, "spin_length", spin_length_SU2);
        
        if (positions_SU2 != nullptr && positions_SU2->size() == lattice_size_SU2) {
            write_positions(meta_SU2_group_, *positions_SU2, lattice_size_SU2);
        }
        
        // SU3 metadata
        write_scalar_attribute(meta_SU3_group_, "lattice_size", lattice_size_SU3);
        write_scalar_attribute(meta_SU3_group_, "spin_dim", spin_dim_SU3);
        write_scalar_attribute(meta_SU3_group_, "n_atoms", n_atoms_SU3);
        write_double_attribute(meta_SU3_group_, "spin_length", spin_length_SU3);
        
        if (positions_SU3 != nullptr && positions_SU3->size() == lattice_size_SU3) {
            write_positions(meta_SU3_group_, *positions_SU3, lattice_size_SU3);
        }
        
        // Create expandable datasets for SU2
        create_trajectory_datasets(traj_SU2_group_, lattice_size_SU2, spin_dim_SU2,
                                   times_SU2_ds_, mag_af_SU2_ds_, mag_loc_SU2_ds_, spins_SU2_ds_);
        
        // Create expandable datasets for SU3
        create_trajectory_datasets(traj_SU3_group_, lattice_size_SU3, spin_dim_SU3,
                                   times_SU3_ds_, mag_af_SU3_ds_, mag_loc_SU3_ds_, spins_SU3_ds_);
    }
    
    /**
     * Write a single time step directly from flat state array
     */
    void write_flat_step(double time, 
                        const SpinVector& mag_af_SU2, const SpinVector& mag_loc_SU2,
                        const SpinVector& mag_af_SU3, const SpinVector& mag_loc_SU3,
                        const double* flat_state) {
        // Write SU2 data
        write_step_data(times_SU2_ds_, mag_af_SU2_ds_, mag_loc_SU2_ds_, spins_SU2_ds_,
                       time, mag_af_SU2, mag_loc_SU2, flat_state, 
                       lattice_size_SU2_, spin_dim_SU2_, 0);
        
        // Write SU3 data (offset in flat array)
        size_t offset_SU3 = lattice_size_SU2_ * spin_dim_SU2_;
        write_step_data(times_SU3_ds_, mag_af_SU3_ds_, mag_loc_SU3_ds_, spins_SU3_ds_,
                       time, mag_af_SU3, mag_loc_SU3, flat_state + offset_SU3,
                       lattice_size_SU3_, spin_dim_SU3_, 0);
        
        current_step_++;
    }
    
    void close() {
        times_SU2_ds_.close();
        mag_af_SU2_ds_.close();
        mag_loc_SU2_ds_.close();
        spins_SU2_ds_.close();
        
        times_SU3_ds_.close();
        mag_af_SU3_ds_.close();
        mag_loc_SU3_ds_.close();
        spins_SU3_ds_.close();
        
        traj_SU2_group_.close();
        traj_SU3_group_.close();
        meta_SU2_group_.close();
        meta_SU3_group_.close();
        meta_global_group_.close();
        file_.close();
    }
    
    ~HDF5MixedMDWriter() {
        try {
            if (file_.getId() > 0) {
                close();
            }
        } catch (...) {}
    }
    
private:
    void create_trajectory_datasets(H5::Group& group, size_t lattice_size, size_t spin_dim,
                                   H5::DataSet& times_ds, H5::DataSet& mag_af_ds,
                                   H5::DataSet& mag_loc_ds, H5::DataSet& spins_ds) {
        // Times dataset
        hsize_t time_dims[1] = {0};
        hsize_t time_maxdims[1] = {H5S_UNLIMITED};
        hsize_t time_chunk[1] = {1000};
        H5::DataSpace time_space(1, time_dims, time_maxdims);
        H5::DSetCreatPropList time_prop;
        time_prop.setChunk(1, time_chunk);
        time_prop.setDeflate(6);
        times_ds = group.createDataSet("times", H5::PredType::NATIVE_DOUBLE, time_space, time_prop);
        
        // Magnetization datasets [n_steps, spin_dim]
        hsize_t mag_dims[2] = {0, spin_dim};
        hsize_t mag_maxdims[2] = {H5S_UNLIMITED, spin_dim};
        hsize_t mag_chunk[2] = {100, spin_dim};
        H5::DataSpace mag_space(2, mag_dims, mag_maxdims);
        H5::DSetCreatPropList mag_prop;
        mag_prop.setChunk(2, mag_chunk);
        mag_prop.setDeflate(6);
        mag_af_ds = group.createDataSet("magnetization_antiferro", H5::PredType::NATIVE_DOUBLE, mag_space, mag_prop);
        mag_loc_ds = group.createDataSet("magnetization_local", H5::PredType::NATIVE_DOUBLE, mag_space, mag_prop);
        
        // Spins dataset [n_steps, n_sites, spin_dim]
        hsize_t spin_dims[3] = {0, lattice_size, spin_dim};
        hsize_t spin_maxdims[3] = {H5S_UNLIMITED, lattice_size, spin_dim};
        hsize_t spin_chunk[3] = {1, lattice_size, spin_dim};
        H5::DataSpace spin_space(3, spin_dims, spin_maxdims);
        H5::DSetCreatPropList spin_prop;
        spin_prop.setChunk(3, spin_chunk);
        spin_prop.setDeflate(6);
        spins_ds = group.createDataSet("spins", H5::PredType::NATIVE_DOUBLE, spin_space, spin_prop);
    }
    
    void write_step_data(H5::DataSet& times_ds, H5::DataSet& mag_af_ds, H5::DataSet& mag_loc_ds, H5::DataSet& spins_ds,
                        double time, const SpinVector& mag_af, const SpinVector& mag_loc,
                        const double* flat_spins, size_t lattice_size, size_t spin_dim, size_t step_offset) {
        hsize_t new_size[3];
        size_t step = current_step_ - step_offset;
        
        // Write time
        new_size[0] = step + 1;
        times_ds.extend(new_size);
        H5::DataSpace time_fspace = times_ds.getSpace();
        hsize_t time_offset[1] = {step};
        hsize_t time_count[1] = {1};
        time_fspace.selectHyperslab(H5S_SELECT_SET, time_count, time_offset);
        H5::DataSpace time_mspace(1, time_count);
        times_ds.write(&time, H5::PredType::NATIVE_DOUBLE, time_mspace, time_fspace);
        
        // Write magnetization antiferro
        new_size[0] = step + 1;
        new_size[1] = spin_dim;
        mag_af_ds.extend(new_size);
        H5::DataSpace mag_af_fspace = mag_af_ds.getSpace();
        hsize_t mag_offset[2] = {step, 0};
        hsize_t mag_count[2] = {1, spin_dim};
        mag_af_fspace.selectHyperslab(H5S_SELECT_SET, mag_count, mag_offset);
        H5::DataSpace mag_mspace(2, mag_count);
        std::vector<double> mag_af_data(spin_dim);
        for (size_t i = 0; i < spin_dim; ++i) mag_af_data[i] = mag_af(i);
        mag_af_ds.write(mag_af_data.data(), H5::PredType::NATIVE_DOUBLE, mag_mspace, mag_af_fspace);
        
        // Write magnetization local
        mag_loc_ds.extend(new_size);
        H5::DataSpace mag_loc_fspace = mag_loc_ds.getSpace();
        mag_loc_fspace.selectHyperslab(H5S_SELECT_SET, mag_count, mag_offset);
        std::vector<double> mag_loc_data(spin_dim);
        for (size_t i = 0; i < spin_dim; ++i) mag_loc_data[i] = mag_loc(i);
        mag_loc_ds.write(mag_loc_data.data(), H5::PredType::NATIVE_DOUBLE, mag_mspace, mag_loc_fspace);
        
        // Write spins directly from flat array
        new_size[0] = step + 1;
        new_size[1] = lattice_size;
        new_size[2] = spin_dim;
        spins_ds.extend(new_size);
        H5::DataSpace spin_fspace = spins_ds.getSpace();
        hsize_t spin_offset[3] = {step, 0, 0};
        hsize_t spin_count[3] = {1, lattice_size, spin_dim};
        spin_fspace.selectHyperslab(H5S_SELECT_SET, spin_count, spin_offset);
        H5::DataSpace spin_mspace(3, spin_count);
        spins_ds.write(flat_spins, H5::PredType::NATIVE_DOUBLE, spin_mspace, spin_fspace);
    }
    
    void write_positions(H5::Group& group, const std::vector<Eigen::Vector3d>& positions, size_t lattice_size) {
        hsize_t pos_dims[2] = {lattice_size, 3};
        H5::DataSpace pos_space(2, pos_dims);
        H5::DataSet pos_dataset = group.createDataSet(
            "positions", H5::PredType::NATIVE_DOUBLE, pos_space);
        
        std::vector<double> pos_data(lattice_size * 3);
        for (size_t i = 0; i < lattice_size; ++i) {
            pos_data[i * 3 + 0] = positions[i](0);
            pos_data[i * 3 + 1] = positions[i](1);
            pos_data[i * 3 + 2] = positions[i](2);
        }
        pos_dataset.write(pos_data.data(), H5::PredType::NATIVE_DOUBLE);
    }
    
    void write_scalar_attribute(H5::Group& group, const std::string& name, hsize_t value) {
        H5::DataSpace attr_space(H5S_SCALAR);
        H5::Attribute attr = group.createAttribute(name, H5::PredType::NATIVE_HSIZE, attr_space);
        attr.write(H5::PredType::NATIVE_HSIZE, &value);
    }
    
    void write_double_attribute(H5::Group& group, const std::string& name, double value) {
        H5::DataSpace attr_space(H5S_SCALAR);
        H5::Attribute attr = group.createAttribute(name, H5::PredType::NATIVE_DOUBLE, attr_space);
        attr.write(H5::PredType::NATIVE_DOUBLE, &value);
    }
    
    void write_string_attribute(H5::Group& group, const std::string& name, const std::string& value) {
        H5::StrType str_type(H5::PredType::C_S1, value.size() + 1);
        H5::DataSpace attr_space(H5S_SCALAR);
        H5::Attribute attr = group.createAttribute(name, str_type, attr_space);
        attr.write(str_type, value.c_str());
    }
    
    std::string filename_;
    size_t lattice_size_SU2_, spin_dim_SU2_;
    size_t lattice_size_SU3_, spin_dim_SU3_;
    size_t current_step_;
    
    H5::H5File file_;
    H5::Group traj_SU2_group_, traj_SU3_group_;
    H5::Group meta_SU2_group_, meta_SU3_group_, meta_global_group_;
    H5::DataSet times_SU2_ds_, mag_af_SU2_ds_, mag_loc_SU2_ds_, spins_SU2_ds_;
    H5::DataSet times_SU3_ds_, mag_af_SU3_ds_, mag_loc_SU3_ds_, spins_SU3_ds_;
};

/**
 * HDF5 Writer for pump-probe spectroscopy data
 * 
 * File structure:
 * /metadata/
 *   - Comprehensive experimental parameters
 * /reference/
 *   - times [n_times]
 *   - M_antiferro [n_times, spin_dim]
 *   - M_local [n_times, spin_dim]
 * /tau_scan/
 *   - tau_values [n_tau]
 *   - tau_0/, tau_1/, ... (each contains M1 and M01 trajectories)
 */
class HDF5PumpProbeWriter {
public:
    HDF5PumpProbeWriter(const std::string& filename,
                       // Lattice parameters
                       size_t lattice_size, size_t spin_dim, size_t n_atoms,
                       size_t dim1, size_t dim2, size_t dim3, float spin_length,
                       // Pulse parameters
                       double pulse_amp, double pulse_width, double pulse_freq,
                       // Time evolution
                       double T_start, double T_end, double T_step,
                       const std::string& integration_method,
                       // Delay scan
                       double tau_start, double tau_end, double tau_step,
                       // Ground state info
                       double ground_state_energy,
                       const SpinVector& ground_magnetization,
                       double Temp_start, double Temp_end, size_t n_anneal,
                       bool T_zero_quench, size_t quench_sweeps,
                       // Optional data
                       const std::vector<SpinVector>* pulse_field_direction = nullptr,
                       const std::vector<Eigen::Vector3d>* site_positions = nullptr)
        : filename_(filename), spin_dim_(spin_dim)
    {
        file_ = H5::H5File(filename, H5F_ACC_TRUNC);
        
        // Create groups
        metadata_group_ = file_.createGroup("/metadata");
        reference_group_ = file_.createGroup("/reference");
        tau_scan_group_ = file_.createGroup("/tau_scan");
        
        // Write metadata
        write_metadata(lattice_size, spin_dim, n_atoms, dim1, dim2, dim3, spin_length,
                      pulse_amp, pulse_width, pulse_freq,
                      T_start, T_end, T_step, integration_method,
                      tau_start, tau_end, tau_step,
                      ground_state_energy, ground_magnetization,
                      Temp_start, Temp_end, n_anneal, T_zero_quench, quench_sweeps,
                      pulse_field_direction, site_positions);
        
        // Compute and store tau values
        int n_tau = static_cast<int>(std::abs((tau_end - tau_start) / tau_step)) + 1;
        std::vector<double> tau_vals(n_tau);
        for (int i = 0; i < n_tau; ++i) {
            tau_vals[i] = tau_start + i * tau_step;
        }
        
        hsize_t tau_dims[1] = {static_cast<hsize_t>(n_tau)};
        H5::DataSpace tau_space(1, tau_dims);
        H5::DataSet tau_dataset = tau_scan_group_.createDataSet(
            "tau_values", H5::PredType::NATIVE_DOUBLE, tau_space);
        tau_dataset.write(tau_vals.data(), H5::PredType::NATIVE_DOUBLE);
    }
    
    /**
     * Write reference trajectory (M0 - single pulse at t=0)
     */
    void write_reference_trajectory(
        const std::vector<std::pair<double, std::array<SpinVector, 3>>>& trajectory) {
        
        size_t n_times = trajectory.size();
        
        // Write times [n_times]
        std::vector<double> times(n_times);
        for (size_t i = 0; i < n_times; ++i) {
            times[i] = trajectory[i].first;
        }
        hsize_t time_dims[1] = {n_times};
        H5::DataSpace time_space(1, time_dims);
        H5::DataSet time_ds = reference_group_.createDataSet(
            "times", H5::PredType::NATIVE_DOUBLE, time_space);
        time_ds.write(times.data(), H5::PredType::NATIVE_DOUBLE);
        
        // Write M_antiferro [n_times, spin_dim]
        write_magnetization_dataset(reference_group_, "M_antiferro", trajectory, n_times, 0);
        
        // Write M_local [n_times, spin_dim]
        write_magnetization_dataset(reference_group_, "M_local", trajectory, n_times, 1);
        
        // Write M_global [n_times, spin_dim]
        write_magnetization_dataset(reference_group_, "M_global", trajectory, n_times, 2);
    }
    
    /**
     * Write delay-dependent trajectories for a specific tau
     */
    void write_tau_trajectory(int tau_index, double tau_value,
                             const std::vector<std::pair<double, std::array<SpinVector, 3>>>& M1_trajectory,
                             const std::vector<std::pair<double, std::array<SpinVector, 3>>>& M01_trajectory) {
        
        // Create group for this tau
        std::string tau_group_name = "/tau_scan/tau_" + std::to_string(tau_index);
        H5::Group tau_group = file_.createGroup(tau_group_name);
        
        // Write tau value as attribute
        H5::DataSpace attr_space(H5S_SCALAR);
        H5::Attribute tau_attr = tau_group.createAttribute(
            "tau_value", H5::PredType::NATIVE_DOUBLE, attr_space);
        tau_attr.write(H5::PredType::NATIVE_DOUBLE, &tau_value);
        
        size_t n_times = M1_trajectory.size();
        
        // Write M1 trajectories
        write_magnetization_dataset(tau_group, "M1_antiferro", M1_trajectory, n_times, 0);
        write_magnetization_dataset(tau_group, "M1_local", M1_trajectory, n_times, 1);
        write_magnetization_dataset(tau_group, "M1_global", M1_trajectory, n_times, 2);
        
        // Write M01 trajectories
        write_magnetization_dataset(tau_group, "M01_antiferro", M01_trajectory, n_times, 0);
        write_magnetization_dataset(tau_group, "M01_local", M01_trajectory, n_times, 1);
        write_magnetization_dataset(tau_group, "M01_global", M01_trajectory, n_times, 2);
    }
    
    void close() {
        metadata_group_.close();
        reference_group_.close();
        tau_scan_group_.close();
        file_.close();
    }
    
    ~HDF5PumpProbeWriter() {
        try {
            if (file_.getId() > 0) close();
        } catch (...) {}
    }
    
private:
    void write_metadata(size_t lattice_size, size_t spin_dim, size_t n_atoms,
                       size_t dim1, size_t dim2, size_t dim3, float spin_length,
                       double pulse_amp, double pulse_width, double pulse_freq,
                       double T_start, double T_end, double T_step,
                       const std::string& integration_method,
                       double tau_start, double tau_end, double tau_step,
                       double ground_state_energy, const SpinVector& ground_mag,
                       double Temp_start, double Temp_end, size_t n_anneal,
                       bool T_zero_quench, size_t quench_sweeps,
                       const std::vector<SpinVector>* pulse_field_direction,
                       const std::vector<Eigen::Vector3d>* site_positions) {
        
        // Timestamp
        std::time_t now = std::time(nullptr);
        char time_str[100];
        std::strftime(time_str, sizeof(time_str), "%Y-%m-%dT%H:%M:%S", std::localtime(&now));
        
        write_string_attr(metadata_group_, "creation_time", std::string(time_str));
        write_string_attr(metadata_group_, "experiment_type", "pump_probe_spectroscopy");
        write_string_attr(metadata_group_, "code_version", "ClassicalSpin_Cpp v1.0");
        write_string_attr(metadata_group_, "file_format", "HDF5_PumpProbe_v1.0");
        
        // Lattice parameters
        write_scalar_attr(metadata_group_, "lattice_size", lattice_size);
        write_scalar_attr(metadata_group_, "spin_dim", spin_dim);
        write_scalar_attr(metadata_group_, "n_atoms", n_atoms);
        write_scalar_attr(metadata_group_, "dim1", dim1);
        write_scalar_attr(metadata_group_, "dim2", dim2);
        write_scalar_attr(metadata_group_, "dim3", dim3);
        write_double_attr(metadata_group_, "spin_length", spin_length);
        
        // Pulse parameters
        write_double_attr(metadata_group_, "pulse_amp", pulse_amp);
        write_double_attr(metadata_group_, "pulse_width", pulse_width);
        write_double_attr(metadata_group_, "pulse_freq", pulse_freq);
        
        // Time evolution
        write_double_attr(metadata_group_, "T_start", T_start);
        write_double_attr(metadata_group_, "T_end", T_end);
        write_double_attr(metadata_group_, "T_step", T_step);
        write_string_attr(metadata_group_, "integration_method", integration_method);
        
        // Delay scan
        write_double_attr(metadata_group_, "tau_start", tau_start);
        write_double_attr(metadata_group_, "tau_end", tau_end);
        write_double_attr(metadata_group_, "tau_step", tau_step);
        int n_tau = static_cast<int>(std::abs((tau_end - tau_start) / tau_step)) + 1;
        write_int_attr(metadata_group_, "tau_steps", n_tau);
        
        // Ground state
        write_double_attr(metadata_group_, "ground_state_energy", ground_state_energy);
        write_double_attr(metadata_group_, "ground_state_magnetization", ground_mag.norm());
        write_double_attr(metadata_group_, "Temp_start", Temp_start);
        write_double_attr(metadata_group_, "Temp_end", Temp_end);
        write_scalar_attr(metadata_group_, "n_anneal", n_anneal);
        write_string_attr(metadata_group_, "T_zero_quench", T_zero_quench ? "true" : "false");
        if (T_zero_quench) {
            write_scalar_attr(metadata_group_, "quench_sweeps", quench_sweeps);
        }
        
        // Optional: pulse field direction
        if (pulse_field_direction && !pulse_field_direction->empty()) {
            write_vector_field(metadata_group_, "pulse_field_direction", 
                             *pulse_field_direction, n_atoms, spin_dim);
        }
        
        // Optional: site positions
        if (site_positions && !site_positions->empty()) {
            write_positions(metadata_group_, *site_positions, lattice_size);
        }
    }
    
    void write_magnetization_dataset(H5::Group& group, const std::string& name,
                                     const std::vector<std::pair<double, std::array<SpinVector, 3>>>& trajectory,
                                     size_t n_times, size_t mag_index) {
        std::vector<double> mag_data(n_times * spin_dim_);
        for (size_t t = 0; t < n_times; ++t) {
            const SpinVector& mag = trajectory[t].second[mag_index];
            for (size_t d = 0; d < spin_dim_; ++d) {
                mag_data[t * spin_dim_ + d] = mag(d);
            }
        }
        
        hsize_t dims[2] = {n_times, spin_dim_};
        H5::DataSpace dataspace(2, dims);
        H5::DataSet dataset = group.createDataSet(name, H5::PredType::NATIVE_DOUBLE, dataspace);
        dataset.write(mag_data.data(), H5::PredType::NATIVE_DOUBLE);
    }
    
    void write_vector_field(H5::Group& group, const std::string& name,
                           const std::vector<SpinVector>& field, size_t n_atoms, size_t spin_dim) {
        std::vector<double> field_data(n_atoms * spin_dim);
        for (size_t i = 0; i < n_atoms && i < field.size(); ++i) {
            for (size_t d = 0; d < spin_dim; ++d) {
                field_data[i * spin_dim + d] = field[i](d);
            }
        }
        hsize_t dims[2] = {n_atoms, spin_dim};
        H5::DataSpace dataspace(2, dims);
        H5::DataSet dataset = group.createDataSet(name, H5::PredType::NATIVE_DOUBLE, dataspace);
        dataset.write(field_data.data(), H5::PredType::NATIVE_DOUBLE);
    }
    
    void write_positions(H5::Group& group, const std::vector<Eigen::Vector3d>& positions, size_t lattice_size) {
        std::vector<double> pos_data(lattice_size * 3);
        for (size_t i = 0; i < lattice_size; ++i) {
            pos_data[i * 3 + 0] = positions[i](0);
            pos_data[i * 3 + 1] = positions[i](1);
            pos_data[i * 3 + 2] = positions[i](2);
        }
        hsize_t dims[2] = {lattice_size, 3};
        H5::DataSpace dataspace(2, dims);
        H5::DataSet dataset = group.createDataSet("positions", H5::PredType::NATIVE_DOUBLE, dataspace);
        dataset.write(pos_data.data(), H5::PredType::NATIVE_DOUBLE);
    }
    
    void write_scalar_attr(H5::Group& group, const std::string& name, hsize_t value) {
        H5::DataSpace attr_space(H5S_SCALAR);
        H5::Attribute attr = group.createAttribute(name, H5::PredType::NATIVE_HSIZE, attr_space);
        attr.write(H5::PredType::NATIVE_HSIZE, &value);
    }
    
    void write_int_attr(H5::Group& group, const std::string& name, int value) {
        H5::DataSpace attr_space(H5S_SCALAR);
        H5::Attribute attr = group.createAttribute(name, H5::PredType::NATIVE_INT, attr_space);
        attr.write(H5::PredType::NATIVE_INT, &value);
    }
    
    void write_double_attr(H5::Group& group, const std::string& name, double value) {
        H5::DataSpace attr_space(H5S_SCALAR);
        H5::Attribute attr = group.createAttribute(name, H5::PredType::NATIVE_DOUBLE, attr_space);
        attr.write(H5::PredType::NATIVE_DOUBLE, &value);
    }
    
    void write_string_attr(H5::Group& group, const std::string& name, const std::string& value) {
        H5::StrType str_type(H5::PredType::C_S1, value.size() + 1);
        H5::DataSpace attr_space(H5S_SCALAR);
        H5::Attribute attr = group.createAttribute(name, str_type, attr_space);
        attr.write(str_type, value.c_str());
    }
    
    std::string filename_;
    size_t spin_dim_;
    H5::H5File file_;
    H5::Group metadata_group_;
    H5::Group reference_group_;
    H5::Group tau_scan_group_;
};

/**
 * HDF5 Writer for mixed lattice pump-probe spectroscopy
 * Maintains separate SU(2) and SU(3) data with consistent nomenclature
 */
class HDF5MixedPumpProbeWriter {
public:
    HDF5MixedPumpProbeWriter(const std::string& filename,
                            // Lattice parameters
                            size_t lattice_size_SU2, size_t spin_dim_SU2, size_t n_atoms_SU2,
                            size_t lattice_size_SU3, size_t spin_dim_SU3, size_t n_atoms_SU3,
                            size_t dim1, size_t dim2, size_t dim3,
                            float spin_length_SU2, float spin_length_SU3,
                            // Pulse parameters
                            double pulse_amp_SU2, double pulse_width_SU2, double pulse_freq_SU2,
                            double pulse_amp_SU3, double pulse_width_SU3, double pulse_freq_SU3,
                            // Time evolution
                            double T_start, double T_end, double T_step,
                            const std::string& integration_method,
                            // Delay scan
                            double tau_start, double tau_end, double tau_step,
                            // Ground state info
                            double ground_state_energy,
                            const SpinVector& ground_mag_SU2, const SpinVector& ground_mag_SU3,
                            double Temp_start, double Temp_end, size_t n_anneal,
                            bool T_zero_quench, size_t quench_sweeps,
                            // Optional data
                            const std::vector<SpinVector>* pulse_field_SU2 = nullptr,
                            const std::vector<SpinVector>* pulse_field_SU3 = nullptr,
                            const std::vector<Eigen::Vector3d>* positions_SU2 = nullptr,
                            const std::vector<Eigen::Vector3d>* positions_SU3 = nullptr)
        : filename_(filename), spin_dim_SU2_(spin_dim_SU2), spin_dim_SU3_(spin_dim_SU3)
    {
        file_ = H5::H5File(filename, H5F_ACC_TRUNC);
        
        // Create groups
        metadata_group_ = file_.createGroup("/metadata");
        reference_group_ = file_.createGroup("/reference");
        tau_scan_group_ = file_.createGroup("/tau_scan");
        
        // Write metadata
        write_metadata(lattice_size_SU2, spin_dim_SU2, n_atoms_SU2,
                      lattice_size_SU3, spin_dim_SU3, n_atoms_SU3,
                      dim1, dim2, dim3, spin_length_SU2, spin_length_SU3,
                      pulse_amp_SU2, pulse_width_SU2, pulse_freq_SU2,
                      pulse_amp_SU3, pulse_width_SU3, pulse_freq_SU3,
                      T_start, T_end, T_step, integration_method,
                      tau_start, tau_end, tau_step,
                      ground_state_energy, ground_mag_SU2, ground_mag_SU3,
                      Temp_start, Temp_end, n_anneal, T_zero_quench, quench_sweeps,
                      pulse_field_SU2, pulse_field_SU3, positions_SU2, positions_SU3);
        
        // Compute and store tau values
        int n_tau = static_cast<int>(std::abs((tau_end - tau_start) / tau_step)) + 1;
        std::vector<double> tau_vals(n_tau);
        for (int i = 0; i < n_tau; ++i) {
            tau_vals[i] = tau_start + i * tau_step;
        }
        
        hsize_t tau_dims[1] = {static_cast<hsize_t>(n_tau)};
        H5::DataSpace tau_space(1, tau_dims);
        H5::DataSet tau_dataset = tau_scan_group_.createDataSet(
            "tau_values", H5::PredType::NATIVE_DOUBLE, tau_space);
        tau_dataset.write(tau_vals.data(), H5::PredType::NATIVE_DOUBLE);
    }
    
    /**
     * Write reference trajectory (M0 - single pulse at t=0)
     * Mixed trajectory format: (time, ([M_af_SU2, M_loc_SU2, M_glob_SU2], [M_af_SU3, M_loc_SU3, M_glob_SU3]))
     */
    void write_reference_trajectory(
        const std::vector<std::pair<double, std::pair<std::array<SpinVector, 3>, 
                                                       std::array<SpinVector, 3>>>>& trajectory) {
        
        size_t n_times = trajectory.size();
        
        // Write times [n_times]
        std::vector<double> times(n_times);
        for (size_t i = 0; i < n_times; ++i) {
            times[i] = trajectory[i].first;
        }
        hsize_t time_dims[1] = {n_times};
        H5::DataSpace time_space(1, time_dims);
        H5::DataSet time_ds = reference_group_.createDataSet(
            "times", H5::PredType::NATIVE_DOUBLE, time_space);
        time_ds.write(times.data(), H5::PredType::NATIVE_DOUBLE);
        
        // Write SU(2) magnetizations
        write_mixed_magnetization(reference_group_, "M_antiferro_SU2", trajectory, n_times, spin_dim_SU2_, 0, true);
        write_mixed_magnetization(reference_group_, "M_local_SU2", trajectory, n_times, spin_dim_SU2_, 1, true);
        write_mixed_magnetization(reference_group_, "M_global_SU2", trajectory, n_times, spin_dim_SU2_, 2, true);
        
        // Write SU(3) magnetizations
        write_mixed_magnetization(reference_group_, "M_antiferro_SU3", trajectory, n_times, spin_dim_SU3_, 0, false);
        write_mixed_magnetization(reference_group_, "M_local_SU3", trajectory, n_times, spin_dim_SU3_, 1, false);
        write_mixed_magnetization(reference_group_, "M_global_SU3", trajectory, n_times, spin_dim_SU3_, 2, false);
    }
    
    /**
     * Write delay-dependent trajectories for a specific tau
     */
    void write_tau_trajectory(int tau_index, double tau_value,
                             const std::vector<std::pair<double, std::pair<std::array<SpinVector, 3>, 
                                                                           std::array<SpinVector, 3>>>>& M1_trajectory,
                             const std::vector<std::pair<double, std::pair<std::array<SpinVector, 3>, 
                                                                           std::array<SpinVector, 3>>>>& M01_trajectory) {
        
        // Create group for this tau
        std::string tau_group_name = "/tau_scan/tau_" + std::to_string(tau_index);
        H5::Group tau_group = file_.createGroup(tau_group_name);
        
        // Write tau value as attribute
        H5::DataSpace attr_space(H5S_SCALAR);
        H5::Attribute tau_attr = tau_group.createAttribute(
            "tau_value", H5::PredType::NATIVE_DOUBLE, attr_space);
        tau_attr.write(H5::PredType::NATIVE_DOUBLE, &tau_value);
        
        size_t n_times = M1_trajectory.size();
        
        // Write M1 trajectories (SU2 and SU3)
        write_mixed_magnetization(tau_group, "M1_antiferro_SU2", M1_trajectory, n_times, spin_dim_SU2_, 0, true);
        write_mixed_magnetization(tau_group, "M1_local_SU2", M1_trajectory, n_times, spin_dim_SU2_, 1, true);
        write_mixed_magnetization(tau_group, "M1_global_SU2", M1_trajectory, n_times, spin_dim_SU2_, 2, true);
        write_mixed_magnetization(tau_group, "M1_antiferro_SU3", M1_trajectory, n_times, spin_dim_SU3_, 0, false);
        write_mixed_magnetization(tau_group, "M1_local_SU3", M1_trajectory, n_times, spin_dim_SU3_, 1, false);
        write_mixed_magnetization(tau_group, "M1_global_SU3", M1_trajectory, n_times, spin_dim_SU3_, 2, false);
        
        // Write M01 trajectories (SU2 and SU3)
        write_mixed_magnetization(tau_group, "M01_antiferro_SU2", M01_trajectory, n_times, spin_dim_SU2_, 0, true);
        write_mixed_magnetization(tau_group, "M01_local_SU2", M01_trajectory, n_times, spin_dim_SU2_, 1, true);
        write_mixed_magnetization(tau_group, "M01_global_SU2", M01_trajectory, n_times, spin_dim_SU2_, 2, true);
        write_mixed_magnetization(tau_group, "M01_antiferro_SU3", M01_trajectory, n_times, spin_dim_SU3_, 0, false);
        write_mixed_magnetization(tau_group, "M01_local_SU3", M01_trajectory, n_times, spin_dim_SU3_, 1, false);
        write_mixed_magnetization(tau_group, "M01_global_SU3", M01_trajectory, n_times, spin_dim_SU3_, 2, false);
    }
    
    void close() {
        metadata_group_.close();
        reference_group_.close();
        tau_scan_group_.close();
        file_.close();
    }
    
    ~HDF5MixedPumpProbeWriter() {
        try {
            if (file_.getId() > 0) close();
        } catch (...) {}
    }
    
private:
    void write_metadata(size_t lattice_size_SU2, size_t spin_dim_SU2, size_t n_atoms_SU2,
                       size_t lattice_size_SU3, size_t spin_dim_SU3, size_t n_atoms_SU3,
                       size_t dim1, size_t dim2, size_t dim3,
                       float spin_length_SU2, float spin_length_SU3,
                       double pulse_amp_SU2, double pulse_width_SU2, double pulse_freq_SU2,
                       double pulse_amp_SU3, double pulse_width_SU3, double pulse_freq_SU3,
                       double T_start, double T_end, double T_step,
                       const std::string& integration_method,
                       double tau_start, double tau_end, double tau_step,
                       double ground_state_energy,
                       const SpinVector& ground_mag_SU2, const SpinVector& ground_mag_SU3,
                       double Temp_start, double Temp_end, size_t n_anneal,
                       bool T_zero_quench, size_t quench_sweeps,
                       const std::vector<SpinVector>* pulse_field_SU2,
                       const std::vector<SpinVector>* pulse_field_SU3,
                       const std::vector<Eigen::Vector3d>* positions_SU2,
                       const std::vector<Eigen::Vector3d>* positions_SU3) {
        
        // Timestamp
        std::time_t now = std::time(nullptr);
        char time_str[100];
        std::strftime(time_str, sizeof(time_str), "%Y-%m-%dT%H:%M:%S", std::localtime(&now));
        
        write_string_attr(metadata_group_, "creation_time", std::string(time_str));
        write_string_attr(metadata_group_, "experiment_type", "pump_probe_spectroscopy_mixed");
        write_string_attr(metadata_group_, "code_version", "ClassicalSpin_Cpp v1.0");
        write_string_attr(metadata_group_, "file_format", "HDF5_PumpProbe_Mixed_v1.0");
        
        // Lattice parameters - SU(2)
        write_scalar_attr(metadata_group_, "lattice_size_SU2", lattice_size_SU2);
        write_scalar_attr(metadata_group_, "spin_dim_SU2", spin_dim_SU2);
        write_scalar_attr(metadata_group_, "n_atoms_SU2", n_atoms_SU2);
        write_double_attr(metadata_group_, "spin_length_SU2", spin_length_SU2);
        
        // Lattice parameters - SU(3)
        write_scalar_attr(metadata_group_, "lattice_size_SU3", lattice_size_SU3);
        write_scalar_attr(metadata_group_, "spin_dim_SU3", spin_dim_SU3);
        write_scalar_attr(metadata_group_, "n_atoms_SU3", n_atoms_SU3);
        write_double_attr(metadata_group_, "spin_length_SU3", spin_length_SU3);
        
        // Common lattice dimensions
        write_scalar_attr(metadata_group_, "dim1", dim1);
        write_scalar_attr(metadata_group_, "dim2", dim2);
        write_scalar_attr(metadata_group_, "dim3", dim3);
        
        // Pulse parameters - SU(2)
        write_double_attr(metadata_group_, "pulse_amp_SU2", pulse_amp_SU2);
        write_double_attr(metadata_group_, "pulse_width_SU2", pulse_width_SU2);
        write_double_attr(metadata_group_, "pulse_freq_SU2", pulse_freq_SU2);
        
        // Pulse parameters - SU(3)
        write_double_attr(metadata_group_, "pulse_amp_SU3", pulse_amp_SU3);
        write_double_attr(metadata_group_, "pulse_width_SU3", pulse_width_SU3);
        write_double_attr(metadata_group_, "pulse_freq_SU3", pulse_freq_SU3);
        
        // Time evolution
        write_double_attr(metadata_group_, "T_start", T_start);
        write_double_attr(metadata_group_, "T_end", T_end);
        write_double_attr(metadata_group_, "T_step", T_step);
        write_string_attr(metadata_group_, "integration_method", integration_method);
        
        // Delay scan
        write_double_attr(metadata_group_, "tau_start", tau_start);
        write_double_attr(metadata_group_, "tau_end", tau_end);
        write_double_attr(metadata_group_, "tau_step", tau_step);
        int n_tau = static_cast<int>(std::abs((tau_end - tau_start) / tau_step)) + 1;
        write_int_attr(metadata_group_, "tau_steps", n_tau);
        
        // Ground state
        write_double_attr(metadata_group_, "ground_state_energy", ground_state_energy);
        write_double_attr(metadata_group_, "ground_state_magnetization_SU2", ground_mag_SU2.norm());
        write_double_attr(metadata_group_, "ground_state_magnetization_SU3", ground_mag_SU3.norm());
        write_double_attr(metadata_group_, "Temp_start", Temp_start);
        write_double_attr(metadata_group_, "Temp_end", Temp_end);
        write_scalar_attr(metadata_group_, "n_anneal", n_anneal);
        write_string_attr(metadata_group_, "T_zero_quench", T_zero_quench ? "true" : "false");
        if (T_zero_quench) {
            write_scalar_attr(metadata_group_, "quench_sweeps", quench_sweeps);
        }
        
        // Optional: pulse field directions
        if (pulse_field_SU2 && !pulse_field_SU2->empty()) {
            write_vector_field(metadata_group_, "pulse_field_direction_SU2", 
                             *pulse_field_SU2, n_atoms_SU2, spin_dim_SU2);
        }
        if (pulse_field_SU3 && !pulse_field_SU3->empty()) {
            write_vector_field(metadata_group_, "pulse_field_direction_SU3", 
                             *pulse_field_SU3, n_atoms_SU3, spin_dim_SU3);
        }
        
        // Optional: site positions
        if (positions_SU2 && !positions_SU2->empty()) {
            write_positions(metadata_group_, "positions_SU2", *positions_SU2, lattice_size_SU2);
        }
        if (positions_SU3 && !positions_SU3->empty()) {
            write_positions(metadata_group_, "positions_SU3", *positions_SU3, lattice_size_SU3);
        }
    }
    
    void write_mixed_magnetization(H5::Group& group, const std::string& name,
                                   const std::vector<std::pair<double, std::pair<std::array<SpinVector, 3>, 
                                                                                 std::array<SpinVector, 3>>>>& trajectory,
                                   size_t n_times, size_t spin_dim, size_t mag_index, bool use_SU2) {
        std::vector<double> mag_data(n_times * spin_dim);
        for (size_t t = 0; t < n_times; ++t) {
            const SpinVector& mag = use_SU2 ? 
                trajectory[t].second.first[mag_index] : 
                trajectory[t].second.second[mag_index];
            for (size_t d = 0; d < spin_dim; ++d) {
                mag_data[t * spin_dim + d] = mag(d);
            }
        }
        
        hsize_t dims[2] = {n_times, spin_dim};
        H5::DataSpace dataspace(2, dims);
        H5::DataSet dataset = group.createDataSet(name, H5::PredType::NATIVE_DOUBLE, dataspace);
        dataset.write(mag_data.data(), H5::PredType::NATIVE_DOUBLE);
    }
    
    void write_vector_field(H5::Group& group, const std::string& name,
                           const std::vector<SpinVector>& field, size_t n_atoms, size_t spin_dim) {
        std::vector<double> field_data(n_atoms * spin_dim);
        for (size_t i = 0; i < n_atoms && i < field.size(); ++i) {
            for (size_t d = 0; d < spin_dim; ++d) {
                field_data[i * spin_dim + d] = field[i](d);
            }
        }
        hsize_t dims[2] = {n_atoms, spin_dim};
        H5::DataSpace dataspace(2, dims);
        H5::DataSet dataset = group.createDataSet(name, H5::PredType::NATIVE_DOUBLE, dataspace);
        dataset.write(field_data.data(), H5::PredType::NATIVE_DOUBLE);
    }
    
    void write_positions(H5::Group& group, const std::string& name,
                        const std::vector<Eigen::Vector3d>& positions, size_t lattice_size) {
        std::vector<double> pos_data(lattice_size * 3);
        for (size_t i = 0; i < lattice_size; ++i) {
            pos_data[i * 3 + 0] = positions[i](0);
            pos_data[i * 3 + 1] = positions[i](1);
            pos_data[i * 3 + 2] = positions[i](2);
        }
        hsize_t dims[2] = {lattice_size, 3};
        H5::DataSpace dataspace(2, dims);
        H5::DataSet dataset = group.createDataSet(name, H5::PredType::NATIVE_DOUBLE, dataspace);
        dataset.write(pos_data.data(), H5::PredType::NATIVE_DOUBLE);
    }
    
    void write_scalar_attr(H5::Group& group, const std::string& name, hsize_t value) {
        H5::DataSpace attr_space(H5S_SCALAR);
        H5::Attribute attr = group.createAttribute(name, H5::PredType::NATIVE_HSIZE, attr_space);
        attr.write(H5::PredType::NATIVE_HSIZE, &value);
    }
    
    void write_int_attr(H5::Group& group, const std::string& name, int value) {
        H5::DataSpace attr_space(H5S_SCALAR);
        H5::Attribute attr = group.createAttribute(name, H5::PredType::NATIVE_INT, attr_space);
        attr.write(H5::PredType::NATIVE_INT, &value);
    }
    
    void write_double_attr(H5::Group& group, const std::string& name, double value) {
        H5::DataSpace attr_space(H5S_SCALAR);
        H5::Attribute attr = group.createAttribute(name, H5::PredType::NATIVE_DOUBLE, attr_space);
        attr.write(H5::PredType::NATIVE_DOUBLE, &value);
    }
    
    void write_string_attr(H5::Group& group, const std::string& name, const std::string& value) {
        H5::StrType str_type(H5::PredType::C_S1, value.size() + 1);
        H5::DataSpace attr_space(H5S_SCALAR);
        H5::Attribute attr = group.createAttribute(name, str_type, attr_space);
        attr.write(str_type, value.c_str());
    }
    
    std::string filename_;
    size_t spin_dim_SU2_, spin_dim_SU3_;
    H5::H5File file_;
    H5::Group metadata_group_;
    H5::Group reference_group_;
    H5::Group tau_scan_group_;
};

#endif // HDF5_IO_H
