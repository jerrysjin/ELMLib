//////////////////////////////////////////////////////////////////////////
/**
*   C++ implementation of Extreme Leaning Machine (ELM)
*
*   Fundamental definitions of this ELM lib
*
*   Code developed by Shuo Jin at Dept. of MAE, CUHK, Hong Kong
*   Email: jerry.shuojin@gmail.com. All rights reserved.
*/
//////////////////////////////////////////////////////////////////////////

#ifndef HEADER_ELM_BASE_H
#define HEADER_ELM_BASE_H

#include "elm_macro.h"

/** Specify the path for the header files 'Core' and 'SVD' of Eigen library
*   For example,
*   #include <Eigen/Core>
*   #include <Eigen/SVD>
*/

#include <Eigen/Core>
#include <Eigen/SVD>

#include <vector>
#include <cmath>
#include <string>
#include <iostream>
#include <iomanip>
#include <exception>

namespace elm
{
	/** Sample defining a pair of input and target data
	*   \T  Sample data type
	*   \ID The dimension of input data
	*   \TD The dimension of target data
	*/
	template <class T, size_t ID, size_t TD>
	struct elm_sample
	{
		/** Data array */
		T i_data[ID];
		T t_data[TD];

		/** Default constructor */
		elm_sample()
		{
			for (size_t i = 0; i < ID; ++i)
			{
				i_data[i] = 0;
			}
			for (size_t i = 0; i < TD; ++i)
			{
				t_data[i] = 0;
			}
		}

		/** Copy constructor */
		elm_sample(const elm_sample & _spl)
		{
			for (size_t i = 0; i < ID; ++i)
			{
				i_data[i] = _spl.i_data[i];
			}
			for (size_t i = 0; i < TD; ++i)
			{
				t_data[i] = _spl.t_data[i];
			}
		}

		/** Operator = */
		elm_sample & operator = (const elm_sample & _spl)
		{
			for (size_t i = 0; i < ID; ++i)
			{
				i_data[i] = _spl.i_data[i];
			}
			for (size_t i = 0; i < TD; ++i)
			{
				t_data[i] = _spl.t_data[i];
			}

			return *this;
		}

		/** Output on console */
		void output_on_console() const
		{
			for (size_t i = 0; i < ID; ++i)
			{
				std::cout << std::setprecision(4) << i_data[i] << " ";
			}
			std::cout << "| ";
			for (size_t i = 0; i < TD; ++i)
			{
				std::cout << std::setprecision(4) << t_data[i] << " ";
			}
			std::cout << std::endl;
		}
	};

	/** Exception delivering proper error message */
	struct elm_exception : public std::exception
	{
		std::string message;

		elm_exception() { }

		elm_exception(const char* _msg) : message(_msg) { }

		elm_exception(std::string _msg) : message(_msg) { }

		const char* what() const _NOEXCEPT
		{
			return message.c_str();
		}
	};

	/** Base class for every elm class to prevent any copy operation.
	*   Every elm class is inherited from this base class.
	*/
	class _elm_non_copyable_
	{
	public:
		_elm_non_copyable_() { }

		virtual ~_elm_non_copyable_() { }

	private:
		_elm_non_copyable_(const _elm_non_copyable_ &);

		_elm_non_copyable_ & operator = (const _elm_non_copyable_ &);
	};

	/** Check if the C++ standard version of the compiler is valid */
	const bool elm_cpp_version_check()
	{
		if (__cplusplus > 201100L)
		{
			return true;
		}
		return false;
	}

} // namespace elm

namespace Eigen
{
	template <class T>
	using MatrixXT = Matrix<T, -1, -1, 0, -1, -1>;
} // namespace Eigen

#endif // HEADER_ELM_BASE_H