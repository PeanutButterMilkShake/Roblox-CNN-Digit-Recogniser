-- Activation functions with their derivatives for neural network layers
local activationFunctions = {}

-- Sigmoid: outputs values between 0 and 1, smooth gradient
activationFunctions.Sigmoid = {
	func = function(x)
		return 1 / (1 + math.exp(-x)) -- maps any number to a value between 0 and 1
	end,
	-- Derivative for backpropagation (assumes x is already activated)
	derivative = function(x)
		return x * (1 - x) -- calculates the slope for backpropagation
	end
}

-- Tanh: outputs values between -1 and 1, zero-centered
activationFunctions.Tanh = {
	func = function(x)
		return math.tanh(x) -- maps any number to a value between -1 and 1
	end,
	-- Derivative for backpropagation (assumes x is already activated)
	derivative = function(x)
		return 1 - (x * x) -- calculates the slope for backpropagation
	end
}

-- ReLU: outputs max(0, x), computationally efficient
activationFunctions.ReLU = {
	func = function(x)
		return math.max(0, x) -- just keeps positive numbers and zeros out negatives
	end,
	-- Derivative for backpropagation (assumes x is already activated)
	derivative = function(x)
		return x > 0 and 1 or 0 -- returns 1 for positive numbers, 0 otherwise
	end
}

local NeuralNetworkService = {}
NeuralNetworkService.__index = NeuralNetworkService

-- Xavier/Glorot weight initialization for better convergence
-- Scales weights based on number of inputs and outputs to prevent vanishing/exploding gradients
local function initializeWeights(inputSize, outputSize)
	local weights = {} -- stores the connection strengths between layers
	local scale = math.sqrt(2.0 / (inputSize + outputSize)) -- calculates an appropriate range for initial weights
	for i = 1, inputSize do -- loops through each input neuron
		weights[i] = {} -- creates a row for this neuron's connections
		for j = 1, outputSize do -- loops through each output neuron
			-- Random values in range [-scale, scale]
			weights[i][j] = (math.random() * 2 - 1) * scale -- generates a random starting weight
		end
	end
	return weights -- returns the initialized weight matrix
end

-- Constructor: creates a new neural network
-- layerSizes: array of integers defining neurons per layer (e.g., {784, 128, 64, 10})
-- activationName: string specifying activation function ("Sigmoid", "Tanh", or "ReLU")
function NeuralNetworkService.new(layerSizes, activationName)
	local self = setmetatable({}, NeuralNetworkService) -- creates a new network instance
	self.layers = {} -- stores all the network layers
	self.numLayers = #layerSizes -- counts how many layers we have
	self.layerSizes = layerSizes -- stores the layer sizes

	-- Select and validate activation function
	local activation = activationFunctions[activationName] -- looks up the requested activation function
	if not activation then -- checks if the function exists
		warn("Unknown activation function: " .. tostring(activationName) .. ", defaulting to Sigmoid") -- prints a warning
		activation = activationFunctions.Sigmoid -- falls back to sigmoid
	end
	self.activation = activation -- stores the activation function

	-- Initialize layers with random weights and zero biases
	-- Each layer connects to the next layer
	for i = 1, self.numLayers - 1 do -- creates connections between consecutive layers
		local layer = { -- builds a layer structure
			weights = initializeWeights(layerSizes[i], layerSizes[i + 1]), -- creates random starting weights
			biases = {}, -- stores bias values
			outputs = {} -- stores neuron outputs during forward pass
		}
		-- Initialize all biases to zero
		for j = 1, layerSizes[i + 1] do -- loops through each neuron in the next layer
			layer.biases[j] = 0 -- sets initial bias to zero
		end
		table.insert(self.layers, layer) -- adds the layer to the network
	end

	return self -- returns the newly created network
end

-- Forward pass: propagate inputs through the network to get predictions
-- inputs: array of input values matching first layer size
-- Returns: array of output values from final layer
function NeuralNetworkService:calculateOutputs(inputs)
	local outputs = inputs -- starts with the input values
	-- Store outputs of each layer for backpropagation
	self.layerOutputs = {inputs} -- saves layer outputs for training

	-- Process each layer
	for i = 1, self.numLayers - 1 do -- passes data through each layer
		local layer = self.layers[i] -- gets the current layer
		local newOutputs = {} -- stores this layer's outputs

		-- Calculate output for each neuron in current layer
		for j = 1, #layer.weights[1] do -- processes each neuron
			-- Start with bias term
			local weightedSum = layer.biases[j] -- starts with the bias value
			-- Add weighted sum of all inputs
			for k = 1, #outputs do -- loops through each input
				weightedSum = weightedSum + outputs[k] * layer.weights[k][j] -- multiplies input by weight and adds to sum
			end
			-- Apply activation function
			newOutputs[j] = self.activation.func(weightedSum) -- applies the activation function
		end

		layer.outputs = newOutputs -- saves this layer's outputs
		table.insert(self.layerOutputs, newOutputs) -- stores for backpropagation
		outputs = newOutputs -- passes outputs to next layer
	end

	return outputs -- returns final layer outputs
end

-- Train the network using mini-batch gradient descent with backpropagation
-- trainingData: array of {data = inputArray, label = classIndex}
-- epochs: number of times to iterate through entire dataset
-- learningRate: step size for weight updates (typical: 0.01 - 0.1)
-- batchSize: number of samples to process before updating weights (default: 1)
-- yieldEvery: yield after processing this many samples to prevent timeout (default: 5)
function NeuralNetworkService:train(trainingData, epochs, learningRate, batchSize, yieldEvery)
	batchSize = batchSize or 1 -- defaults to 1 if not provided
	yieldEvery = yieldEvery or 5 -- defaults to 5 if not provided

	for epoch = 1, epochs do -- repeats training for the specified number of epochs
		local totalError = 0 -- tracks total error for this epoch
		local samplesProcessed = 0 -- counts processed samples
		local batchCount = 0 -- tracks samples in current batch

		-- Shuffle training data each epoch for better generalization
		for i = #trainingData, 2, -1 do -- loops backwards through the data
			local j = math.random(i) -- picks a random index
			trainingData[i], trainingData[j] = trainingData[j], trainingData[i] -- swaps two elements
		end

		-- Initialize gradient accumulators for mini-batch
		local weightUpdates = {} -- accumulates weight gradients
		local biasUpdates = {} -- accumulates bias gradients

		for layerIndex, layer in ipairs(self.layers) do -- sets up accumulators for each layer
			weightUpdates[layerIndex] = {} -- creates weight accumulator
			biasUpdates[layerIndex] = {} -- creates bias accumulator
			for j = 1, #layer.weights do -- loops through input connections
				weightUpdates[layerIndex][j] = {} -- creates row
				for k = 1, #layer.weights[j] do -- loops through output connections
					weightUpdates[layerIndex][j][k] = 0 -- initializes to zero
				end
			end
			for j = 1, #layer.biases do -- loops through biases
				biasUpdates[layerIndex][j] = 0 -- initializes to zero
			end
		end

		-- Process each training sample
		for _, data in ipairs(trainingData) do -- iterates through training examples
			local inputs = data.data -- gets the input data
			local expectedOutputs = {} -- creates target output array

			-- Convert class label to one-hot encoding
			-- e.g., label 3 becomes [0, 0, 1, 0, ...]
			for i = 1, self.layerSizes[self.numLayers] do -- loops through output neurons
				expectedOutputs[i] = 0 -- sets all to zero
			end
			expectedOutputs[data.label] = 1 -- sets correct class to 1

			-- Forward pass: get network predictions
			local outputs = self:calculateOutputs(inputs) -- runs forward pass

			-- Calculate output layer errors using squared error
			local outputErrors = {} -- stores error for each output
			for i = 1, #outputs do -- loops through outputs
				local error = expectedOutputs[i] - outputs[i] -- calculates the difference
				outputErrors[i] = error -- stores the error
				totalError = totalError + error * error -- adds squared error to total
			end

			-- Backpropagation: propagate errors backward through hidden layers
			local layerErrors = {outputErrors} -- starts with output errors
			for i = self.numLayers - 1, 2, -1 do -- works backwards through hidden layers
				local layer = self.layers[i] -- gets current layer
				local prevErrors = {} -- stores errors for previous layer

				-- Calculate error for each neuron in previous layer
				for j = 1, #layer.weights do -- loops through previous layer neurons
					local error = 0 -- initializes error
					-- Sum weighted errors from next layer
					for k = 1, #layer.weights[j] do -- loops through connections forward
						error = error + layer.weights[j][k] * layerErrors[1][k] -- weights error by connection strength
					end
					prevErrors[j] = error -- stores calculated error
				end
				table.insert(layerErrors, 1, prevErrors) -- adds to front of error list
			end

			-- Accumulate gradients for batch update
			for i = 1, self.numLayers - 1 do -- processes each layer
				local layer = self.layers[i] -- gets the layer
				local currentOutputs = self.layerOutputs[i + 1] -- gets this layer's outputs
				local prevOutputs = self.layerOutputs[i] -- gets previous layer's outputs
				local errors = layerErrors[i] -- gets this layer's errors

				-- Update weight gradients
				for j = 1, #layer.weights do -- loops through weight rows
					for k = 1, #layer.weights[j] do -- loops through weight columns
						-- Gradient = error * derivative * input
						local delta = errors[k] * self.activation.derivative(currentOutputs[k]) -- calculates gradient signal
						weightUpdates[i][j][k] = weightUpdates[i][j][k] + delta * prevOutputs[j] -- accumulates gradient
					end
				end

				-- Update bias gradients
				for j = 1, #layer.biases do -- loops through biases
					local delta = errors[j] * self.activation.derivative(currentOutputs[j]) -- calculates gradient signal
					biasUpdates[i][j] = biasUpdates[i][j] + delta -- accumulates gradient
				end
			end

			samplesProcessed += 1 -- increments sample counter
			batchCount += 1 -- increments batch counter

			-- Apply accumulated gradients after processing batch
			if batchCount == batchSize then -- checks if batch is complete
				for i = 1, self.numLayers - 1 do -- updates each layer
					local layer = self.layers[i] -- gets the layer

					-- Update weights using average gradient across batch
					for j = 1, #layer.weights do -- loops through weight rows
						for k = 1, #layer.weights[j] do -- loops through weight columns
							layer.weights[j][k] = layer.weights[j][k] + (learningRate / batchSize) * weightUpdates[i][j][k] -- applies gradient update
							weightUpdates[i][j][k] = 0 -- resets accumulator
						end
					end

					-- Update biases using average gradient across batch
					for j = 1, #layer.biases do -- loops through biases
						layer.biases[j] = layer.biases[j] + (learningRate / batchSize) * biasUpdates[i][j] -- applies gradient update
						biasUpdates[i][j] = 0 -- resets accumulator
					end
				end
				batchCount = 0 -- resets batch counter
			end

			-- Yield periodically to prevent script timeout
			if samplesProcessed % yieldEvery == 0 then -- checks if it's time to yield
				task.wait() -- yields to prevent timeout
			end
		end

		-- Print epoch statistics
		print("Epoch " .. epoch .. "/" .. epochs .. " - Error: " .. string.format("%.4f", totalError)) -- displays progress
		task.wait() -- yields between epochs
	end

	print("Training complete") -- indicates training finished
end

-- Classify an input and return the predicted class label
-- inputs: array of input values
-- Returns: predicted class index, confidence score, all output probabilities
function NeuralNetworkService:classify(inputs)
	local outputs = self:calculateOutputs(inputs) -- runs forward pass
	local maxOutput, predictedLabel = -math.huge, -1 -- initializes with very low values

	-- Find neuron with highest activation (argmax)
	for i = 1, #outputs do -- loops through outputs
		if outputs[i] > maxOutput then -- checks if this output is higher
			maxOutput = outputs[i] -- updates max value
			predictedLabel = i -- updates predicted class
		end
	end

	return predictedLabel, maxOutput, outputs -- returns prediction, confidence, and all outputs
end

-- Get raw output probabilities for all classes
-- inputs: array of input values
-- Returns: array of output activations for each class
function NeuralNetworkService:predictProbs(inputs)
	return self:calculateOutputs(inputs) -- returns all output values
end

-- Evaluate classification accuracy on a test dataset
-- testData: array of {data = inputArray, label = classIndex}
-- Returns: accuracy as decimal between 0 and 1
function NeuralNetworkService:evaluate(testData)
	local correct = 0 -- counts correct predictions
	for _, sample in ipairs(testData) do -- loops through test samples
		local predicted = self:classify(sample.data) -- gets prediction
		if predicted == sample.label then -- checks if prediction is correct
			correct += 1 -- increments correct count
		end
	end
	return correct / #testData -- returns accuracy fraction
end

-- Generate confusion matrix showing prediction patterns
-- testData: array of {data = inputArray, label = classIndex}
-- Returns: 2D matrix where [actual][predicted] = count
function NeuralNetworkService:confusionMatrix(testData)
	local size = self.layerSizes[self.numLayers] -- gets number of classes
	local matrix = {} -- creates matrix table
	
	-- Initialize matrix with zeros
	for i = 1, size do -- loops through rows
		matrix[i] = {} -- creates row
		for j = 1, size do -- loops through columns
			matrix[i][j] = 0 -- initializes to zero
		end
	end

	-- Count predictions for each actual/predicted pair
	for _, sample in ipairs(testData) do -- loops through test samples
		local predicted = self:classify(sample.data) -- gets prediction
		matrix[sample.label][predicted] += 1 -- increments count
	end

	return matrix -- returns completed matrix
end

return NeuralNetworkService -- exports the module
