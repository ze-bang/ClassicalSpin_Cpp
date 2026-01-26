#ifndef HDF5_IO_H
#define HDF5_IO_H

#include <H5Cpp.h>
#include <vector>
#include <string>
#include <Eigen/Dense>
#include <ctime>
#include <sstream>
#include <iomanip>
#include "classical_spin/core/simple_linear_alg.h"

/**
 * Helper function to create HDF5 file with proper serial access properties.
 * This is critical for compatibility with parallel HDF5 libraries (hdf5-mpi).
 * 
 * When the parallel HDF5 library is loaded but we want serial I/O (each MPI rank
 * writing to its own independent file), we must explicitly use the SEC2 (POSIX)
 * file driver which bypasses MPI-IO. Otherwise, the parallel HDF5 library may
 * throw exceptions when creating groups/datasets due to invalid file handles.
 * 
 * @param filename Path to the HDF5 file to create
 * @return H5::H5File object opened for writing
 */
inline H5::H5File create_hdf5_file_serial(const std::string& filename) {
    // Use low-level C API for maximum compatibility with parallel HDF5
    // Set up file access property list with SEC2 (POSIX) driver
    // This forces serial/independent I/O even when linked against parallel HDF5
    hid_t fapl_id = H5Pcreate(H5P_FILE_ACCESS);
    if (fapl_id < 0) {
        throw H5::FileIException("create_hdf5_file_serial", "Failed to create file access property list");
    }
    
    // Use the SEC2 (standard POSIX) driver - this bypasses MPI-IO
    herr_t status = H5Pset_fapl_sec2(fapl_id);
    if (status < 0) {
        H5Pclose(fapl_id);
        throw H5::FileIException("create_hdf5_file_serial", "Failed to set SEC2 file driver");
    }
    
    // Create file with the SEC2 driver
    hid_t file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl_id);
    H5Pclose(fapl_id);
    
    if (file_id < 0) {
        throw H5::FileIException("create_hdf5_file_serial", "Failed to create HDF5 file: " + filename);
    }
    
    // Wrap the C file handle in C++ H5File object
    // The H5File constructor takes ownership of the file_id
    return H5::H5File(file_id);
}

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
 *   - magnetization_global [n_steps, spin_dim]    : Global magnetization (sublattice-frame transformed)
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
        // Create HDF5 file with serial access (compatible with parallel HDF5 library)
        file_ = create_hdf5_file_serial(filename);
        
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
        
        // Magnetization global dataset [n_steps, spin_dim]
        mag_global_dataset_ = trajectory_group_.createDataSet(
            "magnetization_global", H5::PredType::NATIVE_DOUBLE, mag_space, mag_prop);
        
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
                   const SpinVector& mag_global,
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
        
        // Write magnetization global
        mag_global_dataset_.extend(new_size);
        H5::DataSpace mag_global_fspace = mag_global_dataset_.getSpace();
        mag_global_fspace.selectHyperslab(H5S_SELECT_SET, mag_count, mag_offset);
        std::vector<double> mag_global_data(spin_dim_);
        for (size_t i = 0; i < spin_dim_; ++i) {
            mag_global_data[i] = mag_global(i);
        }
        mag_global_dataset_.write(mag_global_data.data(), H5::PredType::NATIVE_DOUBLE, 
                                mag_mspace, mag_global_fspace);
        
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
                        const SpinVector& mag_global,
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
        
        // Write magnetization global
        mag_global_dataset_.extend(new_size);
        H5::DataSpace mag_global_fspace = mag_global_dataset_.getSpace();
        mag_global_fspace.selectHyperslab(H5S_SELECT_SET, mag_count, mag_offset);
        std::vector<double> mag_global_data(spin_dim_);
        for (size_t i = 0; i < spin_dim_; ++i) {
            mag_global_data[i] = mag_global(i);
        }
        mag_global_dataset_.write(mag_global_data.data(), H5::PredType::NATIVE_DOUBLE, 
                                mag_mspace, mag_global_fspace);
        
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
        mag_global_dataset_.close();
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
    H5::DataSet mag_global_dataset_;
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
 *   - magnetization_global [n_steps, spin_dim_SU2]
 * 
 * /trajectory_SU3/
 *   - times [n_steps]
 *   - spins [n_steps, n_sites_SU3, spin_dim_SU3]
 *   - magnetization_antiferro [n_steps, spin_dim_SU3]
 *   - magnetization_local [n_steps, spin_dim_SU3]
 *   - magnetization_global [n_steps, spin_dim_SU3]
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
        // Create HDF5 file with serial access (compatible with parallel HDF5 library)
        file_ = create_hdf5_file_serial(filename);
        
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
                                   times_SU2_ds_, mag_af_SU2_ds_, mag_loc_SU2_ds_, mag_glob_SU2_ds_, spins_SU2_ds_);
        
        // Create expandable datasets for SU3
        create_trajectory_datasets(traj_SU3_group_, lattice_size_SU3, spin_dim_SU3,
                                   times_SU3_ds_, mag_af_SU3_ds_, mag_loc_SU3_ds_, mag_glob_SU3_ds_, spins_SU3_ds_);
    }
    
    /**
     * Write a single time step directly from flat state array
     */
    void write_flat_step(double time, 
                        const SpinVector& mag_af_SU2, const SpinVector& mag_loc_SU2, const SpinVector& mag_glob_SU2,
                        const SpinVector& mag_af_SU3, const SpinVector& mag_loc_SU3, const SpinVector& mag_glob_SU3,
                        const double* flat_state) {
        // Write SU2 data
        write_step_data(times_SU2_ds_, mag_af_SU2_ds_, mag_loc_SU2_ds_, mag_glob_SU2_ds_, spins_SU2_ds_,
                       time, mag_af_SU2, mag_loc_SU2, mag_glob_SU2, flat_state, 
                       lattice_size_SU2_, spin_dim_SU2_, 0);
        
        // Write SU3 data (offset in flat array)
        size_t offset_SU3 = lattice_size_SU2_ * spin_dim_SU2_;
        write_step_data(times_SU3_ds_, mag_af_SU3_ds_, mag_loc_SU3_ds_, mag_glob_SU3_ds_, spins_SU3_ds_,
                       time, mag_af_SU3, mag_loc_SU3, mag_glob_SU3, flat_state + offset_SU3,
                       lattice_size_SU3_, spin_dim_SU3_, 0);
        
        current_step_++;
    }
    
    void close() {
        times_SU2_ds_.close();
        mag_af_SU2_ds_.close();
        mag_loc_SU2_ds_.close();
        mag_glob_SU2_ds_.close();
        spins_SU2_ds_.close();
        
        times_SU3_ds_.close();
        mag_af_SU3_ds_.close();
        mag_loc_SU3_ds_.close();
        mag_glob_SU3_ds_.close();
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
                                   H5::DataSet& mag_loc_ds, H5::DataSet& mag_glob_ds, H5::DataSet& spins_ds) {
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
        mag_glob_ds = group.createDataSet("magnetization_global", H5::PredType::NATIVE_DOUBLE, mag_space, mag_prop);
        
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
    
    void write_step_data(H5::DataSet& times_ds, H5::DataSet& mag_af_ds, H5::DataSet& mag_loc_ds, H5::DataSet& mag_glob_ds, H5::DataSet& spins_ds,
                        double time, const SpinVector& mag_af, const SpinVector& mag_loc, const SpinVector& mag_glob,
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
        
        // Write magnetization global
        mag_glob_ds.extend(new_size);
        H5::DataSpace mag_glob_fspace = mag_glob_ds.getSpace();
        mag_glob_fspace.selectHyperslab(H5S_SELECT_SET, mag_count, mag_offset);
        std::vector<double> mag_glob_data(spin_dim);
        for (size_t i = 0; i < spin_dim; ++i) mag_glob_data[i] = mag_glob(i);
        mag_glob_ds.write(mag_glob_data.data(), H5::PredType::NATIVE_DOUBLE, mag_mspace, mag_glob_fspace);
        
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
    H5::DataSet times_SU2_ds_, mag_af_SU2_ds_, mag_loc_SU2_ds_, mag_glob_SU2_ds_, spins_SU2_ds_;
    H5::DataSet times_SU3_ds_, mag_af_SU3_ds_, mag_loc_SU3_ds_, mag_glob_SU3_ds_, spins_SU3_ds_;
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
        // Create HDF5 file with serial access (compatible with parallel HDF5 library)
        file_ = create_hdf5_file_serial(filename);
        
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
        // Create HDF5 file with serial access (compatible with parallel HDF5 library)
        file_ = create_hdf5_file_serial(filename);
        
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

/**
 * HDF5 Writer for NCTO spin-phonon molecular dynamics trajectories
 * 
 * File structure:
 * /trajectory/
 *   - times [n_steps]                         : Time points
 *   - spins [n_steps, n_sites, 3]             : Full spin configuration
 *   - magnetization [n_steps, 3]              : Total magnetization
 *   - staggered_magnetization [n_steps, 3]    : Staggered (AF) order parameter
 *   - phonons [n_steps, 6]                    : Phonon state (Qx, Qy, Q_R, Vx, Vy, V_R)
 *   - energy [n_steps]                        : Energy per site
 * 
 * /metadata/
 *   - Lattice parameters: lattice_size, dim1, dim2, dim3, spin_length
 *   - Integration parameters: method, dt_initial, T_start, T_end, save_interval
 *   - Phonon parameters: omega_IR, omega_R, gamma_IR, gamma_R, beta, g, Z_star
 *   - Spin-phonon parameters: J1_0, K_0, Gamma_0, J3, lambda_J, lambda_K, lambda_Gamma
 *   - Drive parameters: E0_1, omega_1, t_1, sigma_1, E0_2, omega_2, t_2, sigma_2
 *   - positions [n_sites, 3]: Site positions (optional)
 */
class HDF5NCTOMDWriter {
public:
    HDF5NCTOMDWriter(const std::string& filename, 
                     size_t lattice_size, 
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
          current_step_(0)
    {
        // Create HDF5 file with serial access (compatible with parallel HDF5 library)
        file_ = create_hdf5_file_serial(filename);
        
        // Create groups
        trajectory_group_ = file_.createGroup("/trajectory");
        metadata_group_ = file_.createGroup("/metadata");
        
        // Get current timestamp
        std::time_t now = std::time(nullptr);
        char time_str[100];
        std::strftime(time_str, sizeof(time_str), "%Y-%m-%dT%H:%M:%S", std::localtime(&now));
        
        // Write metadata
        write_scalar_attribute(metadata_group_, "lattice_size", lattice_size);
        write_scalar_attribute(metadata_group_, "dim1", dim1);
        write_scalar_attribute(metadata_group_, "dim2", dim2);
        write_scalar_attribute(metadata_group_, "dim3", dim3);
        write_string_attribute(metadata_group_, "integration_method", method);
        write_double_attribute(metadata_group_, "dt_initial", dt_initial);
        write_double_attribute(metadata_group_, "T_start", T_start);
        write_double_attribute(metadata_group_, "T_end", T_end);
        write_scalar_attribute(metadata_group_, "save_interval", save_interval);
        write_double_attribute(metadata_group_, "spin_length", spin_length);
        write_string_attribute(metadata_group_, "creation_time", std::string(time_str));
        write_string_attribute(metadata_group_, "code_version", "ClassicalSpin_Cpp v1.0");
        write_string_attribute(metadata_group_, "file_format", "HDF5_NCTO_MD_v1.0");
        write_string_attribute(metadata_group_, "lattice_type", "honeycomb_kitaev");
        
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
        
        // Create expandable trajectory datasets
        // Times dataset [n_steps]
        hsize_t time_dims[1] = {0};
        hsize_t time_maxdims[1] = {H5S_UNLIMITED};
        hsize_t time_chunk[1] = {1000};
        H5::DataSpace time_space(1, time_dims, time_maxdims);
        H5::DSetCreatPropList time_prop;
        time_prop.setChunk(1, time_chunk);
        time_prop.setDeflate(6);
        times_dataset_ = trajectory_group_.createDataSet(
            "times", H5::PredType::NATIVE_DOUBLE, time_space, time_prop);
        
        // Magnetization dataset [n_steps, 3]
        hsize_t mag_dims[2] = {0, 3};
        hsize_t mag_maxdims[2] = {H5S_UNLIMITED, 3};
        hsize_t mag_chunk[2] = {100, 3};
        H5::DataSpace mag_space(2, mag_dims, mag_maxdims);
        H5::DSetCreatPropList mag_prop;
        mag_prop.setChunk(2, mag_chunk);
        mag_prop.setDeflate(6);
        magnetization_dataset_ = trajectory_group_.createDataSet(
            "magnetization", H5::PredType::NATIVE_DOUBLE, mag_space, mag_prop);
        
        // Staggered magnetization dataset [n_steps, 3]
        stag_magnetization_dataset_ = trajectory_group_.createDataSet(
            "staggered_magnetization", H5::PredType::NATIVE_DOUBLE, mag_space, mag_prop);
        
        // Phonon dataset [n_steps, 6] (Qx, Qy, Q_R, Vx, Vy, V_R)
        hsize_t ph_dims[2] = {0, 6};
        hsize_t ph_maxdims[2] = {H5S_UNLIMITED, 6};
        hsize_t ph_chunk[2] = {100, 6};
        H5::DataSpace ph_space(2, ph_dims, ph_maxdims);
        H5::DSetCreatPropList ph_prop;
        ph_prop.setChunk(2, ph_chunk);
        ph_prop.setDeflate(6);
        phonons_dataset_ = trajectory_group_.createDataSet(
            "phonons", H5::PredType::NATIVE_DOUBLE, ph_space, ph_prop);
        
        // Energy dataset [n_steps]
        energy_dataset_ = trajectory_group_.createDataSet(
            "energy", H5::PredType::NATIVE_DOUBLE, time_space, time_prop);
        
        // Spins dataset [n_steps, n_sites, 3]
        hsize_t spin_dims[3] = {0, lattice_size, 3};
        hsize_t spin_maxdims[3] = {H5S_UNLIMITED, lattice_size, 3};
        hsize_t spin_chunk[3] = {1, lattice_size, 3};
        H5::DataSpace spin_space(3, spin_dims, spin_maxdims);
        H5::DSetCreatPropList spin_prop;
        spin_prop.setChunk(3, spin_chunk);
        spin_prop.setDeflate(6);
        spins_dataset_ = trajectory_group_.createDataSet(
            "spins", H5::PredType::NATIVE_DOUBLE, spin_space, spin_prop);
    }
    
    /**
     * Write phonon parameters to metadata
     */
    void write_phonon_params(double omega_IR, double omega_R,
                            double gamma_IR, double gamma_R,
                            double beta, double g, double Z_star) {
        H5::Group phonon_group = metadata_group_.createGroup("phonon_params");
        write_double_attribute(phonon_group, "omega_IR", omega_IR);
        write_double_attribute(phonon_group, "omega_R", omega_R);
        write_double_attribute(phonon_group, "gamma_IR", gamma_IR);
        write_double_attribute(phonon_group, "gamma_R", gamma_R);
        write_double_attribute(phonon_group, "beta", beta);
        write_double_attribute(phonon_group, "g", g);
        write_double_attribute(phonon_group, "Z_star", Z_star);
    }
    
    /**
     * Write spin-phonon coupling parameters to metadata
     */
    void write_spin_phonon_params(double J1_0, double K_0, double Gamma_0, double Gammap_0,
                                  double J3, double lambda_J, double lambda_K,
                                  double lambda_Gamma, double lambda_Gammap) {
        H5::Group sp_group = metadata_group_.createGroup("spin_phonon_params");
        write_double_attribute(sp_group, "J1_0", J1_0);
        write_double_attribute(sp_group, "K_0", K_0);
        write_double_attribute(sp_group, "Gamma_0", Gamma_0);
        write_double_attribute(sp_group, "Gammap_0", Gammap_0);
        write_double_attribute(sp_group, "J3", J3);
        write_double_attribute(sp_group, "lambda_J", lambda_J);
        write_double_attribute(sp_group, "lambda_K", lambda_K);
        write_double_attribute(sp_group, "lambda_Gamma", lambda_Gamma);
        write_double_attribute(sp_group, "lambda_Gammap", lambda_Gammap);
    }
    
    /**
     * Write THz drive parameters to metadata
     */
    void write_drive_params(double E0_1, double omega_1, double t_1, double sigma_1, double phi_1, double theta_1,
                           double E0_2, double omega_2, double t_2, double sigma_2, double phi_2, double theta_2) {
        H5::Group drive_group = metadata_group_.createGroup("drive_params");
        write_double_attribute(drive_group, "E0_1", E0_1);
        write_double_attribute(drive_group, "omega_1", omega_1);
        write_double_attribute(drive_group, "t_1", t_1);
        write_double_attribute(drive_group, "sigma_1", sigma_1);
        write_double_attribute(drive_group, "phi_1", phi_1);
        write_double_attribute(drive_group, "theta_1", theta_1);
        write_double_attribute(drive_group, "E0_2", E0_2);
        write_double_attribute(drive_group, "omega_2", omega_2);
        write_double_attribute(drive_group, "t_2", t_2);
        write_double_attribute(drive_group, "sigma_2", sigma_2);
        write_double_attribute(drive_group, "phi_2", phi_2);
        write_double_attribute(drive_group, "theta_2", theta_2);
    }
    
    /**
     * Write a single time step
     * 
     * @param time   Current time
     * @param M      Total magnetization
     * @param Ms     Staggered magnetization
     * @param phonon_state  Phonon state array [Qx, Qy, Q_R, Vx, Vy, V_R]
     * @param energy Energy per site
     * @param spins  Spin configuration [lattice_size, 3]
     */
    void write_step(double time, 
                   const Eigen::Vector3d& M,
                   const Eigen::Vector3d& Ms,
                   const double* phonon_state,
                   double energy,
                   const std::vector<Eigen::Vector3d>& spins) {
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
        
        // Write magnetization
        new_size[0] = current_step_ + 1;
        new_size[1] = 3;
        magnetization_dataset_.extend(new_size);
        H5::DataSpace mag_fspace = magnetization_dataset_.getSpace();
        hsize_t mag_offset[2] = {current_step_, 0};
        hsize_t mag_count[2] = {1, 3};
        mag_fspace.selectHyperslab(H5S_SELECT_SET, mag_count, mag_offset);
        H5::DataSpace mag_mspace(2, mag_count);
        double mag_data[3] = {M(0), M(1), M(2)};
        magnetization_dataset_.write(mag_data, H5::PredType::NATIVE_DOUBLE, mag_mspace, mag_fspace);
        
        // Write staggered magnetization
        stag_magnetization_dataset_.extend(new_size);
        H5::DataSpace stag_fspace = stag_magnetization_dataset_.getSpace();
        stag_fspace.selectHyperslab(H5S_SELECT_SET, mag_count, mag_offset);
        double stag_data[3] = {Ms(0), Ms(1), Ms(2)};
        stag_magnetization_dataset_.write(stag_data, H5::PredType::NATIVE_DOUBLE, mag_mspace, stag_fspace);
        
        // Write phonon state [6]
        new_size[0] = current_step_ + 1;
        new_size[1] = 6;
        phonons_dataset_.extend(new_size);
        H5::DataSpace ph_fspace = phonons_dataset_.getSpace();
        hsize_t ph_offset[2] = {current_step_, 0};
        hsize_t ph_count[2] = {1, 6};
        ph_fspace.selectHyperslab(H5S_SELECT_SET, ph_count, ph_offset);
        H5::DataSpace ph_mspace(2, ph_count);
        phonons_dataset_.write(phonon_state, H5::PredType::NATIVE_DOUBLE, ph_mspace, ph_fspace);
        
        // Write energy
        new_size[0] = current_step_ + 1;
        energy_dataset_.extend(new_size);
        H5::DataSpace e_fspace = energy_dataset_.getSpace();
        hsize_t e_offset[1] = {current_step_};
        hsize_t e_count[1] = {1};
        e_fspace.selectHyperslab(H5S_SELECT_SET, e_count, e_offset);
        H5::DataSpace e_mspace(1, e_count);
        energy_dataset_.write(&energy, H5::PredType::NATIVE_DOUBLE, e_mspace, e_fspace);
        
        // Write spins
        new_size[0] = current_step_ + 1;
        new_size[1] = lattice_size_;
        new_size[2] = 3;
        spins_dataset_.extend(new_size);
        H5::DataSpace spin_fspace = spins_dataset_.getSpace();
        hsize_t spin_offset[3] = {current_step_, 0, 0};
        hsize_t spin_count[3] = {1, lattice_size_, 3};
        spin_fspace.selectHyperslab(H5S_SELECT_SET, spin_count, spin_offset);
        H5::DataSpace spin_mspace(3, spin_count);
        
        std::vector<double> spin_data(lattice_size_ * 3);
        for (size_t i = 0; i < lattice_size_; ++i) {
            spin_data[i * 3 + 0] = spins[i](0);
            spin_data[i * 3 + 1] = spins[i](1);
            spin_data[i * 3 + 2] = spins[i](2);
        }
        spins_dataset_.write(spin_data.data(), H5::PredType::NATIVE_DOUBLE, spin_mspace, spin_fspace);
        
        current_step_++;
    }
    
    /**
     * Write a single time step with flat spin array (zero-copy optimization)
     */
    void write_flat_step(double time, 
                        const Eigen::Vector3d& M,
                        const Eigen::Vector3d& Ms,
                        const double* phonon_state,
                        double energy,
                        const double* flat_spins) {
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
        
        // Write magnetization
        new_size[0] = current_step_ + 1;
        new_size[1] = 3;
        magnetization_dataset_.extend(new_size);
        H5::DataSpace mag_fspace = magnetization_dataset_.getSpace();
        hsize_t mag_offset[2] = {current_step_, 0};
        hsize_t mag_count[2] = {1, 3};
        mag_fspace.selectHyperslab(H5S_SELECT_SET, mag_count, mag_offset);
        H5::DataSpace mag_mspace(2, mag_count);
        double mag_data[3] = {M(0), M(1), M(2)};
        magnetization_dataset_.write(mag_data, H5::PredType::NATIVE_DOUBLE, mag_mspace, mag_fspace);
        
        // Write staggered magnetization
        stag_magnetization_dataset_.extend(new_size);
        H5::DataSpace stag_fspace = stag_magnetization_dataset_.getSpace();
        stag_fspace.selectHyperslab(H5S_SELECT_SET, mag_count, mag_offset);
        double stag_data[3] = {Ms(0), Ms(1), Ms(2)};
        stag_magnetization_dataset_.write(stag_data, H5::PredType::NATIVE_DOUBLE, mag_mspace, stag_fspace);
        
        // Write phonon state [6]
        new_size[0] = current_step_ + 1;
        new_size[1] = 6;
        phonons_dataset_.extend(new_size);
        H5::DataSpace ph_fspace = phonons_dataset_.getSpace();
        hsize_t ph_offset[2] = {current_step_, 0};
        hsize_t ph_count[2] = {1, 6};
        ph_fspace.selectHyperslab(H5S_SELECT_SET, ph_count, ph_offset);
        H5::DataSpace ph_mspace(2, ph_count);
        phonons_dataset_.write(phonon_state, H5::PredType::NATIVE_DOUBLE, ph_mspace, ph_fspace);
        
        // Write energy
        new_size[0] = current_step_ + 1;
        energy_dataset_.extend(new_size);
        H5::DataSpace e_fspace = energy_dataset_.getSpace();
        hsize_t e_offset[1] = {current_step_};
        hsize_t e_count[1] = {1};
        e_fspace.selectHyperslab(H5S_SELECT_SET, e_count, e_offset);
        H5::DataSpace e_mspace(1, e_count);
        energy_dataset_.write(&energy, H5::PredType::NATIVE_DOUBLE, e_mspace, e_fspace);
        
        // Write spins directly from flat array
        new_size[0] = current_step_ + 1;
        new_size[1] = lattice_size_;
        new_size[2] = 3;
        spins_dataset_.extend(new_size);
        H5::DataSpace spin_fspace = spins_dataset_.getSpace();
        hsize_t spin_offset[3] = {current_step_, 0, 0};
        hsize_t spin_count[3] = {1, lattice_size_, 3};
        spin_fspace.selectHyperslab(H5S_SELECT_SET, spin_count, spin_offset);
        H5::DataSpace spin_mspace(3, spin_count);
        spins_dataset_.write(flat_spins, H5::PredType::NATIVE_DOUBLE, spin_mspace, spin_fspace);
        
        current_step_++;
    }
    
    /**
     * Close the file and flush all buffers
     */
    void close() {
        times_dataset_.close();
        magnetization_dataset_.close();
        stag_magnetization_dataset_.close();
        phonons_dataset_.close();
        energy_dataset_.close();
        spins_dataset_.close();
        trajectory_group_.close();
        metadata_group_.close();
        file_.close();
    }
    
    ~HDF5NCTOMDWriter() {
        try {
            if (file_.getId() > 0) {
                close();
            }
        } catch (...) {}
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
    size_t current_step_;
    
    H5::H5File file_;
    H5::Group trajectory_group_;
    H5::Group metadata_group_;
    H5::DataSet times_dataset_;
    H5::DataSet magnetization_dataset_;
    H5::DataSet stag_magnetization_dataset_;
    H5::DataSet phonons_dataset_;
    H5::DataSet energy_dataset_;
    H5::DataSet spins_dataset_;
};

/**
 * HDF5 Writer for Parallel Tempering Monte Carlo simulations
 * 
 * This class provides efficient I/O for parallel tempering by storing
 * all data from a single temperature replica in a single HDF5 file.
 * 
 * File structure:
 * /timeseries/
 *   - energy [n_samples]                          : Energy per site time series
 *   - magnetization [n_samples, spin_dim]         : Total magnetization time series
 *   - sublattice_mag_[α] [n_samples, spin_dim]    : Sublattice α magnetization time series
 * 
 * /observables/
 *   - energy_mean                : Mean energy per site
 *   - energy_error               : Energy error estimate
 *   - specific_heat_mean         : Mean specific heat
 *   - specific_heat_error        : Specific heat error
 *   - sublattice_mag_[α]_mean [spin_dim]  : Mean magnetization of sublattice α
 *   - sublattice_mag_[α]_error [spin_dim] : Error of sublattice α magnetization
 *   - energy_sublattice_[α]_cross_mean [spin_dim]  : Energy-sublattice cross correlation mean
 *   - energy_sublattice_[α]_cross_error [spin_dim] : Energy-sublattice cross correlation error
 * 
 * /metadata/
 *   - temperature           : Simulation temperature
 *   - lattice_size          : Total number of sites
 *   - spin_dim              : Dimension of spin vectors
 *   - n_sublattices         : Number of sublattices (N_atoms)
 *   - n_anneal              : Number of equilibration sweeps
 *   - n_measure             : Number of measurement sweeps
 *   - probe_rate            : Sampling interval
 *   - swap_rate             : Replica exchange attempt rate
 *   - overrelaxation_rate   : Overrelaxation sweep rate
 *   - acceptance_rate       : Metropolis acceptance rate
 *   - swap_acceptance_rate  : Replica exchange acceptance rate
 *   - creation_time         : ISO 8601 timestamp
 *   - code_version          : Version string
 */
class HDF5PTWriter {
public:
    HDF5PTWriter(const std::string& filename,
                 double temperature,
                 size_t lattice_size,
                 size_t spin_dim,
                 size_t n_sublattices,
                 size_t n_samples,
                 size_t n_anneal,
                 size_t n_measure,
                 size_t probe_rate,
                 size_t swap_rate,
                 size_t overrelaxation_rate,
                 double acceptance_rate,
                 double swap_acceptance_rate)
        : filename_(filename),
          lattice_size_(lattice_size),
          spin_dim_(spin_dim),
          n_sublattices_(n_sublattices)
    {
        // Create HDF5 file with serial access (compatible with parallel HDF5 library)
        file_ = create_hdf5_file_serial(filename);
        
        // Create groups
        timeseries_group_ = file_.createGroup("/timeseries");
        observables_group_ = file_.createGroup("/observables");
        metadata_group_ = file_.createGroup("/metadata");
        
        // Get current timestamp
        std::time_t now = std::time(nullptr);
        char time_str[100];
        std::strftime(time_str, sizeof(time_str), "%Y-%m-%dT%H:%M:%S", std::localtime(&now));
        
        // Write metadata
        write_double_attribute(metadata_group_, "temperature", temperature);
        write_scalar_attribute(metadata_group_, "lattice_size", lattice_size);
        write_scalar_attribute(metadata_group_, "spin_dim", spin_dim);
        write_scalar_attribute(metadata_group_, "n_sublattices", n_sublattices);
        write_scalar_attribute(metadata_group_, "n_anneal", n_anneal);
        write_scalar_attribute(metadata_group_, "n_measure", n_measure);
        write_scalar_attribute(metadata_group_, "probe_rate", probe_rate);
        write_scalar_attribute(metadata_group_, "swap_rate", swap_rate);
        write_scalar_attribute(metadata_group_, "overrelaxation_rate", overrelaxation_rate);
        write_double_attribute(metadata_group_, "acceptance_rate", acceptance_rate);
        write_double_attribute(metadata_group_, "swap_acceptance_rate", swap_acceptance_rate);
        write_string_attribute(metadata_group_, "creation_time", std::string(time_str));
        write_string_attribute(metadata_group_, "code_version", "ClassicalSpin_Cpp v1.0");
        write_string_attribute(metadata_group_, "file_format", "HDF5_PT_v1.0");
    }
    
    /**
     * Write time series data
     */
    void write_timeseries(const std::vector<double>& energies,
                         const std::vector<SpinVector>& magnetizations,
                         const std::vector<std::vector<SpinVector>>& sublattice_mags) {
        size_t n_samples = energies.size();
        
        // Write energy time series [n_samples]
        hsize_t energy_dims[1] = {n_samples};
        H5::DataSpace energy_space(1, energy_dims);
        H5::DataSet energy_dataset = timeseries_group_.createDataSet(
            "energy", H5::PredType::NATIVE_DOUBLE, energy_space);
        
        std::vector<double> energy_per_site(n_samples);
        for (size_t i = 0; i < n_samples; ++i) {
            energy_per_site[i] = energies[i] / lattice_size_;
        }
        energy_dataset.write(energy_per_site.data(), H5::PredType::NATIVE_DOUBLE);
        
        // Write magnetization time series [n_samples, spin_dim]
        hsize_t mag_dims[2] = {n_samples, spin_dim_};
        H5::DataSpace mag_space(2, mag_dims);
        H5::DataSet mag_dataset = timeseries_group_.createDataSet(
            "magnetization", H5::PredType::NATIVE_DOUBLE, mag_space);
        
        std::vector<double> mag_data(n_samples * spin_dim_);
        for (size_t i = 0; i < n_samples; ++i) {
            for (size_t d = 0; d < spin_dim_; ++d) {
                mag_data[i * spin_dim_ + d] = magnetizations[i](d);
            }
        }
        mag_dataset.write(mag_data.data(), H5::PredType::NATIVE_DOUBLE);
        
        // Write sublattice magnetization time series for each sublattice
        for (size_t alpha = 0; alpha < n_sublattices_; ++alpha) {
            std::string dataset_name = "sublattice_mag_" + std::to_string(alpha);
            H5::DataSet sub_mag_dataset = timeseries_group_.createDataSet(
                dataset_name, H5::PredType::NATIVE_DOUBLE, mag_space);
            
            std::vector<double> sub_mag_data(n_samples * spin_dim_);
            for (size_t i = 0; i < n_samples; ++i) {
                for (size_t d = 0; d < spin_dim_; ++d) {
                    sub_mag_data[i * spin_dim_ + d] = sublattice_mags[i][alpha](d);
                }
            }
            sub_mag_dataset.write(sub_mag_data.data(), H5::PredType::NATIVE_DOUBLE);
        }
    }
    
    /**
     * Write thermodynamic observables with error estimates
     * Assumes ThermodynamicObservables structure (Observable, VectorObservable)
     */
    void write_observables(double energy_mean, double energy_error,
                          double specific_heat_mean, double specific_heat_error,
                          const std::vector<std::vector<double>>& sublattice_mag_means,
                          const std::vector<std::vector<double>>& sublattice_mag_errors,
                          const std::vector<std::vector<double>>& energy_cross_means,
                          const std::vector<std::vector<double>>& energy_cross_errors) {
        // Write scalar observables
        write_double_dataset(observables_group_, "energy_mean", energy_mean);
        write_double_dataset(observables_group_, "energy_error", energy_error);
        write_double_dataset(observables_group_, "specific_heat_mean", specific_heat_mean);
        write_double_dataset(observables_group_, "specific_heat_error", specific_heat_error);
        
        // Write sublattice magnetization means and errors
        for (size_t alpha = 0; alpha < n_sublattices_; ++alpha) {
            std::string mean_name = "sublattice_mag_" + std::to_string(alpha) + "_mean";
            std::string error_name = "sublattice_mag_" + std::to_string(alpha) + "_error";
            
            write_vector_dataset(observables_group_, mean_name, sublattice_mag_means[alpha]);
            write_vector_dataset(observables_group_, error_name, sublattice_mag_errors[alpha]);
        }
        
        // Write energy-sublattice cross correlations
        for (size_t alpha = 0; alpha < n_sublattices_; ++alpha) {
            std::string mean_name = "energy_sublattice_" + std::to_string(alpha) + "_cross_mean";
            std::string error_name = "energy_sublattice_" + std::to_string(alpha) + "_cross_error";
            
            write_vector_dataset(observables_group_, mean_name, energy_cross_means[alpha]);
            write_vector_dataset(observables_group_, error_name, energy_cross_errors[alpha]);
        }
    }
    
    void close() {
        timeseries_group_.close();
        observables_group_.close();
        metadata_group_.close();
        file_.close();
    }
    
    ~HDF5PTWriter() {
        try {
            if (file_.getId() > 0) {
                close();
            }
        } catch (...) {}
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
    
    void write_double_dataset(H5::Group& group, const std::string& name, double value) {
        H5::DataSpace space(H5S_SCALAR);
        H5::DataSet dataset = group.createDataSet(name, H5::PredType::NATIVE_DOUBLE, space);
        dataset.write(&value, H5::PredType::NATIVE_DOUBLE);
    }
    
    void write_vector_dataset(H5::Group& group, const std::string& name, 
                             const std::vector<double>& data) {
        hsize_t dims[1] = {data.size()};
        H5::DataSpace space(1, dims);
        H5::DataSet dataset = group.createDataSet(name, H5::PredType::NATIVE_DOUBLE, space);
        dataset.write(data.data(), H5::PredType::NATIVE_DOUBLE);
    }
    
    std::string filename_;
    size_t lattice_size_;
    size_t spin_dim_;
    size_t n_sublattices_;
    
    H5::H5File file_;
    H5::Group timeseries_group_;
    H5::Group observables_group_;
    H5::Group metadata_group_;
};

/**
 * HDF5 Writer for Mixed Lattice Parallel Tempering Monte Carlo simulations
 * 
 * This class provides efficient I/O for parallel tempering with mixed SU(2)/SU(3) lattices
 * by storing all data from a single temperature replica in a single HDF5 file.
 * 
 * File structure:
 * /timeseries/
 *   - energy [n_samples]                          : Total energy per site time series
 *   - magnetization_SU2 [n_samples, spin_dim_SU2] : Total SU(2) magnetization time series
 *   - magnetization_SU3 [n_samples, spin_dim_SU3] : Total SU(3) magnetization time series
 *   - sublattice_mag_SU2_[α] [n_samples, spin_dim_SU2]  : SU(2) sublattice α magnetization
 *   - sublattice_mag_SU3_[α] [n_samples, spin_dim_SU3]  : SU(3) sublattice α magnetization
 * 
 * /observables/
 *   - energy_total_mean/error           : Total energy observables
 *   - energy_SU2_mean/error             : SU(2) subsystem energy
 *   - energy_SU3_mean/error             : SU(3) subsystem energy
 *   - specific_heat_mean/error          : Specific heat
 *   - sublattice_mag_SU2_[α]_mean/error : SU(2) sublattice magnetizations
 *   - sublattice_mag_SU3_[α]_mean/error : SU(3) sublattice magnetizations
 *   - energy_sublattice_SU2_[α]_cross_mean/error : Cross-correlations
 *   - energy_sublattice_SU3_[α]_cross_mean/error : Cross-correlations
 * 
 * /metadata/
 *   - temperature, lattice parameters, simulation settings
 */
class HDF5MixedPTWriter {
public:
    HDF5MixedPTWriter(const std::string& filename,
                      double temperature,
                      size_t lattice_size_SU2,
                      size_t lattice_size_SU3,
                      size_t spin_dim_SU2,
                      size_t spin_dim_SU3,
                      size_t n_sublattices_SU2,
                      size_t n_sublattices_SU3,
                      size_t n_samples,
                      size_t n_anneal,
                      size_t n_measure,
                      size_t probe_rate,
                      size_t swap_rate,
                      size_t overrelaxation_rate,
                      double acceptance_rate,
                      double swap_acceptance_rate)
        : filename_(filename),
          lattice_size_SU2_(lattice_size_SU2),
          lattice_size_SU3_(lattice_size_SU3),
          spin_dim_SU2_(spin_dim_SU2),
          spin_dim_SU3_(spin_dim_SU3),
          n_sublattices_SU2_(n_sublattices_SU2),
          n_sublattices_SU3_(n_sublattices_SU3)
    {
        // Create HDF5 file with serial access (compatible with parallel HDF5 library)
        // Each MPI rank writes its own file, so no MPI-parallel HDF5 needed
        file_ = create_hdf5_file_serial(filename);
        
        // Create groups
        timeseries_group_ = file_.createGroup("/timeseries");
        observables_group_ = file_.createGroup("/observables");
        metadata_group_ = file_.createGroup("/metadata");
        
        // Get current timestamp
        std::time_t now = std::time(nullptr);
        char time_str[100];
        std::strftime(time_str, sizeof(time_str), "%Y-%m-%dT%H:%M:%S", std::localtime(&now));
        
        // Write metadata
        write_double_attribute(metadata_group_, "temperature", temperature);
        write_scalar_attribute(metadata_group_, "lattice_size_SU2", lattice_size_SU2);
        write_scalar_attribute(metadata_group_, "lattice_size_SU3", lattice_size_SU3);
        write_scalar_attribute(metadata_group_, "spin_dim_SU2", spin_dim_SU2);
        write_scalar_attribute(metadata_group_, "spin_dim_SU3", spin_dim_SU3);
        write_scalar_attribute(metadata_group_, "n_sublattices_SU2", n_sublattices_SU2);
        write_scalar_attribute(metadata_group_, "n_sublattices_SU3", n_sublattices_SU3);
        write_scalar_attribute(metadata_group_, "n_anneal", n_anneal);
        write_scalar_attribute(metadata_group_, "n_measure", n_measure);
        write_scalar_attribute(metadata_group_, "probe_rate", probe_rate);
        write_scalar_attribute(metadata_group_, "swap_rate", swap_rate);
        write_scalar_attribute(metadata_group_, "overrelaxation_rate", overrelaxation_rate);
        write_double_attribute(metadata_group_, "acceptance_rate", acceptance_rate);
        write_double_attribute(metadata_group_, "swap_acceptance_rate", swap_acceptance_rate);
        write_string_attribute(metadata_group_, "creation_time", std::string(time_str));
        write_string_attribute(metadata_group_, "code_version", "ClassicalSpin_Cpp v1.0");
        write_string_attribute(metadata_group_, "file_format", "HDF5_MixedPT_v1.0");
    }
    
    /**
     * Write time series data for mixed lattice
     */
    void write_timeseries(const std::vector<double>& energies,
                         const std::vector<std::pair<SpinVector, SpinVector>>& magnetizations,
                         const std::vector<std::vector<SpinVector>>& sublattice_mags_SU2,
                         const std::vector<std::vector<SpinVector>>& sublattice_mags_SU3) {
        size_t n_samples = energies.size();
        size_t lattice_size_total = lattice_size_SU2_ + lattice_size_SU3_;
        
        // Write energy time series [n_samples]
        hsize_t energy_dims[1] = {n_samples};
        H5::DataSpace energy_space(1, energy_dims);
        H5::DataSet energy_dataset = timeseries_group_.createDataSet(
            "energy", H5::PredType::NATIVE_DOUBLE, energy_space);
        
        std::vector<double> energy_per_site(n_samples);
        for (size_t i = 0; i < n_samples; ++i) {
            energy_per_site[i] = energies[i] / lattice_size_total;
        }
        energy_dataset.write(energy_per_site.data(), H5::PredType::NATIVE_DOUBLE);
        
        // Write SU(2) magnetization time series [n_samples, spin_dim_SU2]
        hsize_t mag_SU2_dims[2] = {n_samples, spin_dim_SU2_};
        H5::DataSpace mag_SU2_space(2, mag_SU2_dims);
        H5::DataSet mag_SU2_dataset = timeseries_group_.createDataSet(
            "magnetization_SU2", H5::PredType::NATIVE_DOUBLE, mag_SU2_space);
        
        std::vector<double> mag_SU2_data(n_samples * spin_dim_SU2_);
        for (size_t i = 0; i < n_samples; ++i) {
            for (size_t d = 0; d < spin_dim_SU2_; ++d) {
                mag_SU2_data[i * spin_dim_SU2_ + d] = magnetizations[i].first(d);
            }
        }
        mag_SU2_dataset.write(mag_SU2_data.data(), H5::PredType::NATIVE_DOUBLE);
        
        // Write SU(3) magnetization time series [n_samples, spin_dim_SU3]
        hsize_t mag_SU3_dims[2] = {n_samples, spin_dim_SU3_};
        H5::DataSpace mag_SU3_space(2, mag_SU3_dims);
        H5::DataSet mag_SU3_dataset = timeseries_group_.createDataSet(
            "magnetization_SU3", H5::PredType::NATIVE_DOUBLE, mag_SU3_space);
        
        std::vector<double> mag_SU3_data(n_samples * spin_dim_SU3_);
        for (size_t i = 0; i < n_samples; ++i) {
            for (size_t d = 0; d < spin_dim_SU3_; ++d) {
                mag_SU3_data[i * spin_dim_SU3_ + d] = magnetizations[i].second(d);
            }
        }
        mag_SU3_dataset.write(mag_SU3_data.data(), H5::PredType::NATIVE_DOUBLE);
        
        // Write SU(2) sublattice magnetization time series
        for (size_t alpha = 0; alpha < n_sublattices_SU2_; ++alpha) {
            std::string dataset_name = "sublattice_mag_SU2_" + std::to_string(alpha);
            H5::DataSet sub_mag_dataset = timeseries_group_.createDataSet(
                dataset_name, H5::PredType::NATIVE_DOUBLE, mag_SU2_space);
            
            std::vector<double> sub_mag_data(n_samples * spin_dim_SU2_);
            for (size_t i = 0; i < n_samples; ++i) {
                for (size_t d = 0; d < spin_dim_SU2_; ++d) {
                    sub_mag_data[i * spin_dim_SU2_ + d] = sublattice_mags_SU2[i][alpha](d);
                }
            }
            sub_mag_dataset.write(sub_mag_data.data(), H5::PredType::NATIVE_DOUBLE);
        }
        
        // Write SU(3) sublattice magnetization time series
        for (size_t alpha = 0; alpha < n_sublattices_SU3_; ++alpha) {
            std::string dataset_name = "sublattice_mag_SU3_" + std::to_string(alpha);
            H5::DataSet sub_mag_dataset = timeseries_group_.createDataSet(
                dataset_name, H5::PredType::NATIVE_DOUBLE, mag_SU3_space);
            
            std::vector<double> sub_mag_data(n_samples * spin_dim_SU3_);
            for (size_t i = 0; i < n_samples; ++i) {
                for (size_t d = 0; d < spin_dim_SU3_; ++d) {
                    sub_mag_data[i * spin_dim_SU3_ + d] = sublattice_mags_SU3[i][alpha](d);
                }
            }
            sub_mag_dataset.write(sub_mag_data.data(), H5::PredType::NATIVE_DOUBLE);
        }
    }
    
    /**
     * Write thermodynamic observables with error estimates for mixed lattice
     */
    void write_observables(double energy_total_mean, double energy_total_error,
                          double energy_SU2_mean, double energy_SU2_error,
                          double energy_SU3_mean, double energy_SU3_error,
                          double specific_heat_mean, double specific_heat_error,
                          const std::vector<std::vector<double>>& sublattice_mag_SU2_means,
                          const std::vector<std::vector<double>>& sublattice_mag_SU2_errors,
                          const std::vector<std::vector<double>>& sublattice_mag_SU3_means,
                          const std::vector<std::vector<double>>& sublattice_mag_SU3_errors,
                          const std::vector<std::vector<double>>& energy_cross_SU2_means,
                          const std::vector<std::vector<double>>& energy_cross_SU2_errors,
                          const std::vector<std::vector<double>>& energy_cross_SU3_means,
                          const std::vector<std::vector<double>>& energy_cross_SU3_errors) {
        // Write scalar observables
        write_double_dataset(observables_group_, "energy_total_mean", energy_total_mean);
        write_double_dataset(observables_group_, "energy_total_error", energy_total_error);
        write_double_dataset(observables_group_, "energy_SU2_mean", energy_SU2_mean);
        write_double_dataset(observables_group_, "energy_SU2_error", energy_SU2_error);
        write_double_dataset(observables_group_, "energy_SU3_mean", energy_SU3_mean);
        write_double_dataset(observables_group_, "energy_SU3_error", energy_SU3_error);
        write_double_dataset(observables_group_, "specific_heat_mean", specific_heat_mean);
        write_double_dataset(observables_group_, "specific_heat_error", specific_heat_error);
        
        // Write SU(2) sublattice magnetization means and errors
        for (size_t alpha = 0; alpha < n_sublattices_SU2_; ++alpha) {
            std::string mean_name = "sublattice_mag_SU2_" + std::to_string(alpha) + "_mean";
            std::string error_name = "sublattice_mag_SU2_" + std::to_string(alpha) + "_error";
            
            write_vector_dataset(observables_group_, mean_name, sublattice_mag_SU2_means[alpha]);
            write_vector_dataset(observables_group_, error_name, sublattice_mag_SU2_errors[alpha]);
        }
        
        // Write SU(3) sublattice magnetization means and errors
        for (size_t alpha = 0; alpha < n_sublattices_SU3_; ++alpha) {
            std::string mean_name = "sublattice_mag_SU3_" + std::to_string(alpha) + "_mean";
            std::string error_name = "sublattice_mag_SU3_" + std::to_string(alpha) + "_error";
            
            write_vector_dataset(observables_group_, mean_name, sublattice_mag_SU3_means[alpha]);
            write_vector_dataset(observables_group_, error_name, sublattice_mag_SU3_errors[alpha]);
        }
        
        // Write SU(2) energy-sublattice cross correlations
        for (size_t alpha = 0; alpha < n_sublattices_SU2_; ++alpha) {
            std::string mean_name = "energy_sublattice_SU2_" + std::to_string(alpha) + "_cross_mean";
            std::string error_name = "energy_sublattice_SU2_" + std::to_string(alpha) + "_cross_error";
            
            write_vector_dataset(observables_group_, mean_name, energy_cross_SU2_means[alpha]);
            write_vector_dataset(observables_group_, error_name, energy_cross_SU2_errors[alpha]);
        }
        
        // Write SU(3) energy-sublattice cross correlations
        for (size_t alpha = 0; alpha < n_sublattices_SU3_; ++alpha) {
            std::string mean_name = "energy_sublattice_SU3_" + std::to_string(alpha) + "_cross_mean";
            std::string error_name = "energy_sublattice_SU3_" + std::to_string(alpha) + "_cross_error";
            
            write_vector_dataset(observables_group_, mean_name, energy_cross_SU3_means[alpha]);
            write_vector_dataset(observables_group_, error_name, energy_cross_SU3_errors[alpha]);
        }
    }
    
    void close() {
        timeseries_group_.close();
        observables_group_.close();
        metadata_group_.close();
        file_.close();
    }
    
    ~HDF5MixedPTWriter() {
        try {
            if (file_.getId() > 0) {
                close();
            }
        } catch (...) {}
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
    
    void write_double_dataset(H5::Group& group, const std::string& name, double value) {
        H5::DataSpace space(H5S_SCALAR);
        H5::DataSet dataset = group.createDataSet(name, H5::PredType::NATIVE_DOUBLE, space);
        dataset.write(&value, H5::PredType::NATIVE_DOUBLE);
    }
    
    void write_vector_dataset(H5::Group& group, const std::string& name, 
                             const std::vector<double>& data) {
        hsize_t dims[1] = {data.size()};
        H5::DataSpace space(1, dims);
        H5::DataSet dataset = group.createDataSet(name, H5::PredType::NATIVE_DOUBLE, space);
        dataset.write(data.data(), H5::PredType::NATIVE_DOUBLE);
    }
    
    std::string filename_;
    size_t lattice_size_SU2_;
    size_t lattice_size_SU3_;
    size_t spin_dim_SU2_;
    size_t spin_dim_SU3_;
    size_t n_sublattices_SU2_;
    size_t n_sublattices_SU3_;
    
    H5::H5File file_;
    H5::Group timeseries_group_;
    H5::Group observables_group_;
    H5::Group metadata_group_;
};

#endif // HDF5_IO_H
