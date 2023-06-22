/** \namespace problem
* @brief Namespace declaration for modules of type: problem.
*/

/** \file
* @brief Header file for module: SupervisedLearning.
*/

/** \dir problem/supervisedLearning
* @brief Contains code, documentation, and scripts for module: SupervisedLearning.
*/

#pragma once

#include "modules/problem/problem.hpp"

namespace korali
{
namespace problem
{
;

/**
* @brief Class declaration for module: SupervisedLearning.
*/
class SupervisedLearning : public Problem
{
  public: 
  /**
  * @brief Stores the batch size of the training dataset.
  */
   size_t _trainingBatchSize;
  /**
  * @brief Stores the batch size of the testing dataset.
  */
   size_t _testingBatchSize;
  /**
  * @brief Stores the length of the sequence for recurrent neural networks.
  */
   size_t _maxTimesteps;
  /**
  * @brief Provides the input data with layout T*N*IC, where T is the sequence length, N is the batch size and IC is the vector size of the input.
  */
   std::vector<std::vector<std::vector<float>>> _inputData;
  /**
  * @brief Indicates the vector size of the input (IC).
  */
   size_t _inputSize;
  /**
  * @brief Provides the solution for one-step ahead prediction with layout N*OC, where N is the batch size and OC is the vector size of the output.
  */
   std::vector<std::vector<float>> _solutionData;
  /**
  * @brief Indicates the vector size of the output (OC).
  */
   size_t _solutionSize;
  
 
  /**
  * @brief Obtains the entire current state and configuration of the module.
  * @param js JSON object onto which to save the serialized state of the module.
  */
  void getConfiguration(knlohmann::json& js) override;
  /**
  * @brief Sets the entire state and configuration of the module, given a JSON object.
  * @param js JSON object from which to deserialize the state of the module.
  */
  void setConfiguration(knlohmann::json& js) override;
  /**
  * @brief Applies the module's default configuration upon its creation.
  * @param js JSON object containing user configuration. The defaults will not override any currently defined settings.
  */
  void applyModuleDefaults(knlohmann::json& js) override;
  /**
  * @brief Applies the module's default variable configuration to each variable in the Experiment upon creation.
  */
  void applyVariableDefaults() override;
  

  void initialize() override;

  /**
   * @brief Checks whether the input data has the correct shape
   */
  void verifyData();
};

} //problem
} //korali
;
