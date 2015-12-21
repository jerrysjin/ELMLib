//////////////////////////////////////////////////////////////////////////
/**
*   C++ implementation of Extreme Leaning Machine (ELM)
*
*   Basic ELM method with (without) regularization
*
*   Paper reference:
*     Regularized Extreme Learning Machine
*   Link: http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=4938676&tag=1
*
*   Code developed by Shuo Jin at Dept. of MAE, CUHK, Hong Kong
*   Email: jerry.shuojin@gmail.com. All rights reserved.
*/
//////////////////////////////////////////////////////////////////////////

#ifndef HEADER_BASIC_ELM_H
#define HEADER_BASIC_ELM_H

#include "elm_base.h"

#include <limits>
#include <algorithm>
#include <cassert>

namespace elm
{
	/** Different types of activation functions
	*   Other types of activation functions can be customized by users following the below definition.
	*/
	template <class T>
	using elm_activation_function_pointer = const T(*) (Eigen::MatrixXT<T> &, Eigen::MatrixXT<T> &, const T b);

	/** Gaussian RBF */
	template <class T>
	const T elm_activation_function_gaussian_rbf(Eigen::MatrixXT<T> & ai, Eigen::MatrixXT<T> & x, const T bi)
	{
		assert(bi >= 0);

		T sqnorm = (ai - x).squaredNorm();

		return exp(-bi * sqnorm);
	}

	/** Sigmoid */
	template <class T>
	const T elm_activation_function_sigmoid(Eigen::MatrixXT<T> & ai, Eigen::MatrixXT<T> & x, const T bi)
	{
		T dot = ai * x;

		return 1.0 / (1.0 + exp(-dot - bi));
	}

	/** Multiquadric */
	template <class T>
	const T elm_activation_function_multiquadric(Eigen::MatrixXT<T> & ai, Eigen::MatrixXT<T> & x, const T bi)
	{
		T sqnorm = (x - ai).squaredNorm();

		return sqrt(sqnorm + bi * bi);
	}

	/** Add more activation functions here... */


	/** ****************************************************************************************************** */

	/** Class basic_elm
	*   \T  Data type
	*   \ID The Dimension of input data
	*   \TD The Dimension of target data
	*   By default, the Gaussian RBF activation function is used.
	*/
	template <class T, size_t ID, size_t TD>
	class basic_elm : public _elm_non_copyable_
	{
	public:
		basic_elm();

		virtual ~basic_elm();

		/** The indicator controlling the use of regularized ELM scheme */
		enum auto_weighting { enable, disable };

	public: /** Public interface */
		/** Add an input sample for elm training
		*   No exception throw
		*/
		void add_sample(elm_sample<T, ID, TD>);

		/** Call this when all prepared samples are loaded
		*   No exception throw
		*/
		void finish();

		/** Clear all training samples
		*   No exception throw
		*/
		void clear();

		/** Set the number of hidden nodes for ELM training
		*   The hidden nodes number should not be smaller than 2.
		*   No exception throw
		*/
		void set_hidden_nodes_num(const size_t);

		/** Get the number of current hidden nodes
		*   No exception throw
		*/
		const size_t hidden_nodes_num() const;

		/** Set activation function for ELM training
		*   No exception throw
		*/
		void set_activation_function(elm_activation_function_pointer<T> &);

		/** Perform ELM training based on all input samples
		*   \_rglz_factor
		*     The weight to adjust the proportion of empirical risk and structrual risk
		*     If rglz_factor = 0, then this ELM is basic ELM without regularization.
		*     If rglz_factor > 0, then this ELM is regularized ELM, which means
		*     the energy to be minimized is ||H * beta - T||^2 + \_rglz_factor * ||beta||^2.
		*   \_autow_indicator
		*     If autow_indicator = disable, then this regularized ELM is the unweighted version.
		*     If autow_indicator = enable, then this regularized ELM is the weighted version. This
		*     can be enabled if the imported training samples are with noise and outliers and the
		*     number of hidden nodes should be less than the number of samples.
		*   Possible exception throw
		*/
		void train(const T _rglz_factor = 0.0, typename const basic_elm<T, ID, TD>::auto_weighting _autow_indicator = disable);

		/** Predict output sample based on ELM training result
		*   No exception throw
		*/
		void predict(elm_sample<T, ID, TD> &) const;

		/** Get the training result as a matrix
		*   No exception throw
		*/
		Eigen::MatrixXT<T> output_matrix() const;

		/** Export training result to file
		*   Possible exception throw
		*/
		void export_to_file(const char* _file_name);

		/** Import training result from file
		*   Possible exception throw
		*/
		void import_from_file(const char* _file_name);

	private: /** Data members */
		/** The number of hidden nodes */
		size_t belm_hidden_nodes_num;

		/** Training samples for this ELM */
		std::vector< elm_sample<T, ID, TD> > belm_training_samples;

		/** Activation function */
		elm_activation_function_pointer<T> belm_activation_function;

		/** Hidden layer output matrix */
		Eigen::MatrixXT<T> belm_H_matrix;

		/** Randomly generated center vectors */
		Eigen::MatrixXT<T> belm_node_rndm_centers;

		/** Randomly generated bias values for nodes */
		Eigen::MatrixXT<T> belm_node_rndm_bias;

		/** Output weight vectors */
		Eigen::MatrixXT<T> belm_ow_matrix;

		/** Target sample matrix */
		Eigen::MatrixXT<T> belm_target_matrix;

	private: /** Only for internal use */
		/** inv(HtH) * Ht like or Ht * inv(HHt) matrix */
		Eigen::MatrixXT<T> belm_pseudo_inverse_matrix;

		/** Diagonal matrix of weights for regulized ELM */
		Eigen::MatrixXT<T> belm_diagonal_weights;

		/** The mode of pseudo inverse matrix*/
		enum rigde_regression_mode { HTH_mode, HHT_mode } belm_rrmode;

		/** Do initializaiton at the beginning of ELM training
		*   Possible exception throw
		*/
		void belm_initialize();

		/** Compute the unweighted pseudo inverse based on current configurations
		*   No exception throw
		*/
		void belm_compute_pseudo_inverse(const T);

		/** Compute the output weight vectors based on current configurations
		*   Possible exception throw
		*/
		void belm_compute_output_weight_matrix();

		/** Adjust weights for weighted regularized ELM
		*   No exception throw
		*/
		void belm_compute_adaptive_diagonal_weights();

		/** Quick sort with the indexes of elements returned
		*   No exception throw
		*/
		std::vector<size_t> belm_sort_indexes(const std::vector<T>);
	};

	/** Beginning of basic_elm class implementation */
	template <class T, size_t ID, size_t TD>
	basic_elm<T, ID, TD>::basic_elm() :
		belm_hidden_nodes_num(0),
		belm_activation_function(elm_activation_function_gaussian_rbf<T>)
	{

	}

	template <class T, size_t ID, size_t TD>
	basic_elm<T, ID, TD>::~basic_elm()
	{

	}

	template <class T, size_t ID, size_t TD>
	void basic_elm<T, ID, TD>::add_sample(elm_sample<T, ID, TD> _spl)
	{
		belm_training_samples.push_back(_spl);
	}

	template <class T, size_t ID, size_t TD>
	void basic_elm<T, ID, TD>::finish()
	{
		belm_training_samples.shrink_to_fit();
	}

	template <class T, size_t ID, size_t TD>
	void basic_elm<T, ID, TD>::clear()
	{
		belm_training_samples.clear();

		belm_H_matrix.resize(0, 0);
		belm_node_rndm_centers.resize(0, 0);
		belm_node_rndm_bias.resize(0, 0);
		belm_ow_matrix.resize(0, 0);
		belm_target_matrix.resize(0, 0);
		belm_pseudo_inverse_matrix.resize(0, 0);
		belm_diagonal_weights.resize(0, 0);
	}

	template <class T, size_t ID, size_t TD>
	void basic_elm<T, ID, TD>::set_hidden_nodes_num(const size_t _hns_num)
	{
		assert(_hns_num >= 2);

		belm_hidden_nodes_num = _hns_num;
	}

	template <class T, size_t ID, size_t TD>
	const size_t basic_elm<T, ID, TD>::hidden_nodes_num() const
	{
		return belm_hidden_nodes_num;
	}

	template <class T, size_t ID, size_t TD>
	void basic_elm<T, ID, TD>::set_activation_function(elm_activation_function_pointer<T> & _af_ptr)
	{
		belm_activation_function = _af_ptr;
	}

	template <class T, size_t ID, size_t TD>
	void basic_elm<T, ID, TD>::train(const T _rglz_factor, typename const basic_elm<T, ID, TD>::auto_weighting _autow_indicator)
	{
		// Initialization
		belm_initialize();

		// Unweighted regularized ELM	
		belm_compute_pseudo_inverse(_rglz_factor);
		belm_compute_output_weight_matrix();

		// end if auto weighting is disabled
		if (disable == _autow_indicator) return;

		// Weighted regularized ELM 
		belm_compute_adaptive_diagonal_weights();
		belm_compute_pseudo_inverse(_rglz_factor);
		belm_compute_output_weight_matrix();
	}

	template <class T, size_t ID, size_t TD>
	void basic_elm<T, ID, TD>::predict(elm_sample<T, ID, TD> & _spl) const
	{
		Eigen::MatrixXT<T> x = Eigen::Map< Eigen::MatrixXT<T> >(_spl.i_data, 1, ID);

		Eigen::MatrixXT<T> Hx(1, belm_hidden_nodes_num);
		for (size_t i = 0; i < belm_hidden_nodes_num; ++i)
		{
			Eigen::MatrixXT<T> ai(belm_node_rndm_centers.row(i));

			Hx(i) = belm_activation_function(ai, x, belm_node_rndm_bias(i));
		}

		Eigen::MatrixXT<T> y = Hx * belm_ow_matrix;

		Eigen::Map< Eigen::MatrixXT<T> >(_spl.t_data, 1, TD) = y;
	}

	template <class T, size_t ID, size_t TD>
	Eigen::MatrixXT<T> basic_elm<T, ID, TD>::output_matrix() const
	{
		return belm_ow_matrix;
	}

	template <class T, size_t ID, size_t TD>
	void basic_elm<T, ID, TD>::export_to_file(const char* _file_name)
	{
		std::ofstream file(_file_name);
		if (!file.is_open())
		{
			throw elm_exception("ELM: failure in opening file.");
		}

		// training info
		file << "Sample Number = " << belm_training_samples.size() << " Hidden Node Number = " << belm_hidden_nodes_num << std::endl;

		// dimension
		file << belm_ow_matrix.rows() << " " << belm_ow_matrix.cols() << std::endl;

		// training result
		Eigen::IOFormat precision(Eigen::FullPrecision);
		file << belm_ow_matrix << std::endl

			file.close();
	}

	template <class T, size_t ID, size_t TD>
	void basic_elm<T, ID, TD>::import_from_file(const char* _file_name)
	{
		std::ifstream file(_file_name);
		if (!file.is_open())
		{
			throw elm_exception("ELM: failure in opening file.");
		}

		std::string line;

		// ignore 1st line
		file >> line;

		// get matrix size
		file >> line;

		std::stringstream size_stream(line);
		size_t row_count = 0, col_count = 0;

		size_stream >> row_count >> col_count;

		// read matrix
		belm_ow_matrix.resize(row_count, col_count);
		for (size_t i = 0; i < row_count; ++i)
		{
			file >> line;

			std::stringstream data_stream(line);
			for (size_t j = 0; j < col_count; ++j)
			{
				data_stream >> belm_ow_matrix(i, j);
			}
		}

		file.close();
	}

	template <class T, size_t ID, size_t TD>
	void basic_elm<T, ID, TD>::belm_initialize()
	{
		if (0 == belm_training_samples.size())
		{
			throw elm_exception("ELM: no training samples.");
		}

		const size_t sample_size = belm_training_samples.size();

		// basic ELM step 0 - initialize diagonal weights and target matrix
		belm_diagonal_weights.resize(sample_size, sample_size);
		belm_diagonal_weights.setIdentity();

		belm_target_matrix.resize(sample_size, TD);
		for (size_t i = 0; i < belm_training_samples.size(); ++i)
		{
			for (size_t j = 0; j < TD; ++j)
			{
				belm_target_matrix(i, j) = belm_training_samples[i].t_data[j];
			}
		}

		// basic ELM step 1 - randomly generate hidden node parameters
		belm_node_rndm_centers = Eigen::MatrixXT<T>::Random(belm_hidden_nodes_num, ID);

		belm_node_rndm_bias = Eigen::MatrixXT<T>::Random(belm_hidden_nodes_num, 1);

		for (size_t i = 0; i < belm_hidden_nodes_num; ++i)
		{
			belm_node_rndm_bias(i) = abs(belm_node_rndm_bias(i));
		}

		// basic ELM step 2 - calculate the hidden layer output matrix
		belm_H_matrix.resize(sample_size, belm_hidden_nodes_num);
		belm_H_matrix.setZero();

		for (size_t i = 0; i < sample_size; ++i) // row
		{
			for (size_t j = 0; j < belm_hidden_nodes_num; ++j) // col
			{
				Eigen::MatrixXT<T> xi = Eigen::Map< Eigen::MatrixXT<T> >(belm_training_samples[i].i_data, 1, ID);

				Eigen::MatrixXT<T> ai(belm_node_rndm_centers.row(j));

				belm_H_matrix(i, j) = belm_activation_function(ai, xi, belm_node_rndm_bias(j));
			}
		}
	}

	template <class T, size_t ID, size_t TD>
	void basic_elm<T, ID, TD>::belm_compute_pseudo_inverse(const T _rglz_factor)
	{
		const size_t sample_size = belm_training_samples.size();

		// basic ELM step 3 - generate output weights of unweighted ELM
		if (belm_hidden_nodes_num < sample_size)
		{// HTH_mode
			belm_rrmode = HTH_mode;

			Eigen::MatrixXT<T> temp_HTDDH_matrix = belm_H_matrix.transpose() * belm_diagonal_weights * belm_diagonal_weights * belm_H_matrix;

			for (size_t i = 0; i < belm_hidden_nodes_num; ++i)
			{
				temp_HTDDH_matrix(i, i) += _rglz_factor;
			}

			Eigen::JacobiSVD< Eigen::MatrixXT<T> > svd_solver(temp_HTDDH_matrix, Eigen::ComputeThinU | Eigen::ComputeThinV);

#ifdef __ELM_CONSOLE_INFO__
			// output condition number of HHT for reference
			T max_sval = std::numeric_limits<T>::lowest();
			T min_sval = std::numeric_limits<T>::infinity();

			for (size_t i = 0; i < belm_hidden_nodes_num; ++i)
			{
				if (abs(svd_solver.singularValues()(i)) > max_sval)
				{
					max_sval = abs(svd_solver.singularValues()(i));
				}
				if (abs(svd_solver.singularValues()(i)) < min_sval)
				{
					min_sval = abs(svd_solver.singularValues()(i));
				}
			}

			std::cout << "ELM: HTDDH matrix condition number: " << max_sval / min_sval << std::endl;
#endif // __ELM_CONSOLE_INFO__

			// compute inversion of HTH matrix
			belm_pseudo_inverse_matrix = svd_solver.solve(Eigen::MatrixXT<T>::Identity(belm_hidden_nodes_num, belm_hidden_nodes_num));
		}
		else
		{// HHT_mode
			belm_rrmode = HHT_mode;

			Eigen::MatrixXT<T> temp_HHTDD_matrix = belm_H_matrix * belm_H_matrix.transpose() * belm_diagonal_weights * belm_diagonal_weights;

			for (size_t i = 0; i < sample_size; ++i)
			{
				temp_HHTDD_matrix(i, i) += _rglz_factor;
			}

			Eigen::JacobiSVD< Eigen::MatrixXT<T> > svd_solver(temp_HHTDD_matrix, Eigen::ComputeThinU | Eigen::ComputeThinV);

#ifdef __ELM_CONSOLE_INFO__
			// output condition number of HHT for reference
			T max_sval = std::numeric_limits<T>::lowest();
			T min_sval = std::numeric_limits<T>::infinity();

			for (size_t i = 0; i < sample_size; ++i)
			{
				if (abs(svd_solver.singularValues()(i)) > max_sval)
				{
					max_sval = abs(svd_solver.singularValues()(i));
				}
				if (abs(svd_solver.singularValues()(i)) < min_sval)
				{
					min_sval = abs(svd_solver.singularValues()(i));
				}
			}

			std::cout << "ELM: HHTDD matrix condition number: " << max_sval / min_sval << std::endl;
#endif // __ELM_CONSOLE_INFO__

			// compute inversion of HTH matrix	
			belm_pseudo_inverse_matrix = svd_solver.solve(Eigen::MatrixXT<T>::Identity(sample_size, sample_size));
		}
	}

	template <class T, size_t ID, size_t TD>
	void basic_elm<T, ID, TD>::belm_compute_output_weight_matrix()
	{
		Eigen::MatrixXT<T> HTDD_matrix = belm_H_matrix.transpose() * belm_diagonal_weights * belm_diagonal_weights;

		switch (belm_rrmode)
		{
		case HTH_mode:
			belm_ow_matrix = belm_pseudo_inverse_matrix * HTDD_matrix * belm_target_matrix;
			break;

		case HHT_mode:
			belm_ow_matrix = HTDD_matrix * belm_pseudo_inverse_matrix * belm_target_matrix;
			break;

		default:
			throw elm_exception("ELM: fail in computing output weight vectors - rigde regression mode not available.");
		}
	}

	template <class T, size_t ID, size_t TD>
	void basic_elm<T, ID, TD>::belm_compute_adaptive_diagonal_weights()
	{
		const size_t sample_size = belm_training_samples.size();

		// compute residuals
		Eigen::MatrixXT<T> residual_matrix = belm_H_matrix * belm_ow_matrix - belm_target_matrix;

		std::vector<T> residuals(sample_size);
		for (size_t i = 0; i < sample_size; ++i)
		{
			residuals[i] = (residual_matrix.row(i)).norm();
		}

		std::vector<size_t> sorted_idx = belm_sort_indexes(residuals);

		// compute IQR
		T iqr3, iqr1;
		if (1 == sample_size % 2)
		{
			size_t mi = (sample_size - 1) / 2;

			if (1 == mi % 2)
			{
				iqr1 = residuals[sorted_idx[(mi - 1) / 2]];
				iqr3 = residuals[sorted_idx[(sample_size + mi) / 2]];
			}
			else
			{
				iqr1 = 0.5 * (residuals[sorted_idx[mi / 2 - 1]] + residuals[sorted_idx[mi / 2]]);
				iqr3 = 0.5 * (residuals[sorted_idx[3 * mi / 2]] + residuals[sorted_idx[3 * mi / 2 + 1]]);
			}
		}
		else
		{
			size_t hs = sample_size / 2;

			if (1 == hs % 2)
			{
				iqr1 = residuals[sorted_idx[(hs - 1) / 2]];
				iqr3 = residuals[sorted_idx[(hs + sample_size - 1) / 2]];
			}
			else
			{
				iqr1 = 0.5 * (residuals[sorted_idx[hs / 2 - 1]] + residuals[sorted_idx[hs / 2]]);
				iqr3 = 0.5 * (residuals[sorted_idx[3 * hs / 2 - 1]] + residuals[sorted_idx[3 * hs / 2]]);
			}
		}
		T s_hat = (iqr3 - iqr1) / 1.349;

		// update weights in the diagonal matrix
		const T c1 = 2.5, c2 = 3;

		for (size_t i = 0; i < sample_size; ++i)
		{
			const T r = residuals[i] / s_hat;

			if (r > c1 && r <= c2)
			{
				belm_diagonal_weights(i, i) = (c2 - r) / (c2 - c1) + 0.0001;
			}

			if (r > c2)
			{
				belm_diagonal_weights(i, i) = 0.0001;
			}
		}
	}

	template <typename T, size_t ID, size_t TD>
	std::vector<size_t> basic_elm<T, ID, TD>::belm_sort_indexes(const std::vector<T> v)
	{
		std::vector<size_t> idx_vec(v.size());

		for (size_t i = 0; i < idx_vec.size(); ++i)
		{
			idx_vec[i] = i;
		}

		std::sort(idx_vec.begin(), idx_vec.end(), [&v](size_t i_1, size_t i_2) { return v[i_1] < v[i_2]; });

		return idx_vec;
	}

	/** End of basic_elm class implementation */
} // namespace elm

#endif // HEADER_BASIC_ELM_H