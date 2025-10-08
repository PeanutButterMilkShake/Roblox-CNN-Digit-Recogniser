-- Activation functions with their derivatives for neural network layers
local activationFunctions = {}

-- Sigmoid: outputs values between 0 and 1, smooth gradient
activationFunctions.Sigmoid = {
	func = function(x)
		return 1 / (1 + math.exp(-x))
	end,
	-- Derivative for backpropagation (assumes x is already activated)
	derivative = function(x)
		return x * (1 - x)
	end
}

-- Tanh: outputs values between -1 and 1, zero-centered
activationFunctions.Tanh = {
	func = function(x)
		return math.tanh(x)
	end,
	-- Derivative for backpropagation (assumes x is already activated)
	derivative = function(x)
		return 1 - (x * x)
	end
}

-- ReLU: outputs max(0, x), computationally efficient
activationFunctions.ReLU = {
	func = function(x)
		return math.max(0, x)
	end,
	-- Derivative for backpropagation (assumes x is already activated)
	derivative = function(x)
		return x > 0 and 1 or 0
	end
}

local NeuralNetworkService = {}
NeuralNetworkService.__index = NeuralNetworkService

-- Xavier/Glorot weight initialization for better convergence
-- Scales weights based on number of inputs and outputs to prevent vanishing/exploding gradients
local function initializeWeights(inputSize, outputSize)
	local weights = {}
	local scale = math.sqrt(2.0 / (inputSize + outputSize))
	for i = 1, inputSize do
		weights[i] = {}
		for j = 1, outputSize do
			-- Random values in range [-scale, scale]
			weights[i][j] = (math.random() * 2 - 1) * scale
		end
	end
	return weights
end

-- Constructor: creates a new neural network
-- layerSizes: array of integers defining neurons per layer (e.g., {784, 128, 64, 10})
-- activationName: string specifying activation function ("Sigmoid", "Tanh", or "ReLU")
function NeuralNetworkService.new(layerSizes, activationName)
	local self = setmetatable({}, NeuralNetworkService)
	self.layers = {}
	self.numLayers = #layerSizes
	self.layerSizes = layerSizes

	-- Select and validate activation function
	local activation = activationFunctions[activationName]
	if not activation then
		warn("Unknown activation function: " .. tostring(activationName) .. ", defaulting to Sigmoid")
		activation = activationFunctions.Sigmoid
	end
	self.activation = activation

	-- Initialize layers with random weights and zero biases
	-- Each layer connects to the next layer
	for i = 1, self.numLayers - 1 do
		local layer = {
			weights = initializeWeights(layerSizes[i], layerSizes[i + 1]),
			biases = {},
			outputs = {}
		}
		-- Initialize all biases to zero
		for j = 1, layerSizes[i + 1] do
			layer.biases[j] = 0
		end
		table.insert(self.layers, layer)
	end

	return self
end

-- Forward pass: propagate inputs through the network to get predictions
-- inputs: array of input values matching first layer size
-- Returns: array of output values from final layer
function NeuralNetworkService:calculateOutputs(inputs)
	local outputs = inputs
	-- Store outputs of each layer for backpropagation
	self.layerOutputs = {inputs}

	-- Process each layer
	for i = 1, self.numLayers - 1 do
		local layer = self.layers[i]
		local newOutputs = {}

		-- Calculate output for each neuron in current layer
		for j = 1, #layer.weights[1] do
			-- Start with bias term
			local weightedSum = layer.biases[j]
			-- Add weighted sum of all inputs
			for k = 1, #outputs do
				weightedSum = weightedSum + outputs[k] * layer.weights[k][j]
			end
			-- Apply activation function
			newOutputs[j] = self.activation.func(weightedSum)
		end

		layer.outputs = newOutputs
		table.insert(self.layerOutputs, newOutputs)
		outputs = newOutputs
	end

	return outputs
end

-- Train the network using mini-batch gradient descent with backpropagation
-- trainingData: array of {data = inputArray, label = classIndex}
-- epochs: number of times to iterate through entire dataset
-- learningRate: step size for weight updates (typical: 0.01 - 0.1)
-- batchSize: number of samples to process before updating weights (default: 1)
-- yieldEvery: yield after processing this many samples to prevent timeout (default: 5)
function NeuralNetworkService:train(trainingData, epochs, learningRate, batchSize, yieldEvery)
	batchSize = batchSize or 1
	yieldEvery = yieldEvery or 5

	for epoch = 1, epochs do
		local totalError = 0
		local samplesProcessed = 0
		local batchCount = 0

		-- Shuffle training data each epoch for better generalization
		for i = #trainingData, 2, -1 do
			local j = math.random(i)
			trainingData[i], trainingData[j] = trainingData[j], trainingData[i]
		end

		-- Initialize gradient accumulators for mini-batch
		local weightUpdates = {}
		local biasUpdates = {}

		for layerIndex, layer in ipairs(self.layers) do
			weightUpdates[layerIndex] = {}
			biasUpdates[layerIndex] = {}
			for j = 1, #layer.weights do
				weightUpdates[layerIndex][j] = {}
				for k = 1, #layer.weights[j] do
					weightUpdates[layerIndex][j][k] = 0
				end
			end
			for j = 1, #layer.biases do
				biasUpdates[layerIndex][j] = 0
			end
		end

		-- Process each training sample
		for _, data in ipairs(trainingData) do
			local inputs = data.data
			local expectedOutputs = {}

			-- Convert class label to one-hot encoding
			-- e.g., label 3 becomes [0, 0, 1, 0, ...]
			for i = 1, self.layerSizes[self.numLayers] do
				expectedOutputs[i] = 0
			end
			expectedOutputs[data.label] = 1

			-- Forward pass: get network predictions
			local outputs = self:calculateOutputs(inputs)

			-- Calculate output layer errors using squared error
			local outputErrors = {}
			for i = 1, #outputs do
				local error = expectedOutputs[i] - outputs[i]
				outputErrors[i] = error
				totalError = totalError + error * error
			end

			-- Backpropagation: propagate errors backward through hidden layers
			local layerErrors = {outputErrors}
			for i = self.numLayers - 1, 2, -1 do
				local layer = self.layers[i]
				local prevErrors = {}

				-- Calculate error for each neuron in previous layer
				for j = 1, #layer.weights do
					local error = 0
					-- Sum weighted errors from next layer
					for k = 1, #layer.weights[j] do
						error = error + layer.weights[j][k] * layerErrors[1][k]
					end
					prevErrors[j] = error
				end
				table.insert(layerErrors, 1, prevErrors)
			end

			-- Accumulate gradients for batch update
			for i = 1, self.numLayers - 1 do
				local layer = self.layers[i]
				local currentOutputs = self.layerOutputs[i + 1]
				local prevOutputs = self.layerOutputs[i]
				local errors = layerErrors[i]

				-- Update weight gradients
				for j = 1, #layer.weights do
					for k = 1, #layer.weights[j] do
						-- Gradient = error * derivative * input
						local delta = errors[k] * self.activation.derivative(currentOutputs[k])
						weightUpdates[i][j][k] = weightUpdates[i][j][k] + delta * prevOutputs[j]
					end
				end

				-- Update bias gradients
				for j = 1, #layer.biases do
					local delta = errors[j] * self.activation.derivative(currentOutputs[j])
					biasUpdates[i][j] = biasUpdates[i][j] + delta
				end
			end

			samplesProcessed += 1
			batchCount += 1

			-- Apply accumulated gradients after processing batch
			if batchCount == batchSize then
				for i = 1, self.numLayers - 1 do
					local layer = self.layers[i]

					-- Update weights using average gradient across batch
					for j = 1, #layer.weights do
						for k = 1, #layer.weights[j] do
							layer.weights[j][k] = layer.weights[j][k] + (learningRate / batchSize) * weightUpdates[i][j][k]
							weightUpdates[i][j][k] = 0
						end
					end

					-- Update biases using average gradient across batch
					for j = 1, #layer.biases do
						layer.biases[j] = layer.biases[j] + (learningRate / batchSize) * biasUpdates[i][j]
						biasUpdates[i][j] = 0
					end
				end
				batchCount = 0
			end

			-- Yield periodically to prevent script timeout
			if samplesProcessed % yieldEvery == 0 then
				task.wait()
			end
		end

		-- Print epoch statistics
		print("Epoch " .. epoch .. "/" .. epochs .. " - Error: " .. string.format("%.4f", totalError))
		task.wait()
	end

	print("Training complete")
end

-- Classify an input and return the predicted class label
-- inputs: array of input values
-- Returns: predicted class index, confidence score, all output probabilities
function NeuralNetworkService:classify(inputs)
	local outputs = self:calculateOutputs(inputs)
	local maxOutput, predictedLabel = -math.huge, -1

	-- Find neuron with highest activation (argmax)
	for i = 1, #outputs do
		if outputs[i] > maxOutput then
			maxOutput = outputs[i]
			predictedLabel = i
		end
	end

	return predictedLabel, maxOutput, outputs
end

-- Get raw output probabilities for all classes
-- inputs: array of input values
-- Returns: array of output activations for each class
function NeuralNetworkService:predictProbs(inputs)
	return self:calculateOutputs(inputs)
end

-- Evaluate classification accuracy on a test dataset
-- testData: array of {data = inputArray, label = classIndex}
-- Returns: accuracy as decimal between 0 and 1
function NeuralNetworkService:evaluate(testData)
	local correct = 0
	for _, sample in ipairs(testData) do
		local predicted = self:classify(sample.data)
		if predicted == sample.label then
			correct += 1
		end
	end
	return correct / #testData
end

-- Generate confusion matrix showing prediction patterns
-- testData: array of {data = inputArray, label = classIndex}
-- Returns: 2D matrix where [actual][predicted] = count
function NeuralNetworkService:confusionMatrix(testData)
	local size = self.layerSizes[self.numLayers]
	local matrix = {}
	
	-- Initialize matrix with zeros
	for i = 1, size do
		matrix[i] = {}
		for j = 1, size do
			matrix[i][j] = 0
		end
	end

	-- Count predictions for each actual/predicted pair
	for _, sample in ipairs(testData) do
		local predicted = self:classify(sample.data)
		matrix[sample.label][predicted] += 1
	end

	return matrix
end

return NeuralNetworkService
