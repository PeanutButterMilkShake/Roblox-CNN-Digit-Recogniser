-- Activation functions with their derivatives
local activationFunctions = {}

activationFunctions.Sigmoid = {
	func = function(x)
		return 1 / (1 + math.exp(-x))
	end,
	derivative = function(x)
		return x * (1 - x)
	end
}

activationFunctions.Tanh = {
	func = function(x)
		return math.tanh(x)
	end,
	derivative = function(x)
		return 1 - (x * x)
	end
}

activationFunctions.ReLU = {
	func = function(x)
		return math.max(0, x)
	end,
	derivative = function(x)
		return x > 0 and 1 or 0
	end
}

local NeuralNetworkService = {}
NeuralNetworkService.__index = NeuralNetworkService

-- Xavier weight initialization for better convergence
local function initializeWeights(inputSize, outputSize)
	local weights = {}
	local scale = math.sqrt(2.0 / (inputSize + outputSize))
	for i = 1, inputSize do
		weights[i] = {}
		for j = 1, outputSize do
			weights[i][j] = (math.random() * 2 - 1) * scale
		end
	end
	return weights
end

-- Constructor: creates network with specified layer sizes and activation function
function NeuralNetworkService.new(layerSizes, activationName)
	local self = setmetatable({}, NeuralNetworkService)
	self.layers = {}
	self.numLayers = #layerSizes
	self.layerSizes = layerSizes

	-- Select activation function
	local activation = activationFunctions[activationName]
	if not activation then
		warn("Unknown activation function: " .. tostring(activationName) .. ", defaulting to Sigmoid")
		activation = activationFunctions.Sigmoid
	end
	self.activation = activation

	-- Initialize layers with weights and biases
	for i = 1, self.numLayers - 1 do
		local layer = {
			weights = initializeWeights(layerSizes[i], layerSizes[i + 1]),
			biases = {},
			outputs = {}
		}
		for j = 1, layerSizes[i + 1] do
			layer.biases[j] = 0
		end
		table.insert(self.layers, layer)
	end

	return self
end

-- Forward pass: propagate inputs through network
function NeuralNetworkService:calculateOutputs(inputs)
	local outputs = inputs
	self.layerOutputs = {inputs}

	for i = 1, self.numLayers - 1 do
		local layer = self.layers[i]
		local newOutputs = {}

		for j = 1, #layer.weights[1] do
			local weightedSum = layer.biases[j]
			for k = 1, #outputs do
				weightedSum = weightedSum + outputs[k] * layer.weights[k][j]
			end
			newOutputs[j] = self.activation.func(weightedSum)
		end

		layer.outputs = newOutputs
		table.insert(self.layerOutputs, newOutputs)
		outputs = newOutputs
	end

	return outputs
end

-- Train network using mini-batch gradient descent
function NeuralNetworkService:train(trainingData, epochs, learningRate, batchSize, yieldEvery)
	batchSize = batchSize or 1
	yieldEvery = yieldEvery or 5

	for epoch = 1, epochs do
		local totalError = 0
		local samplesProcessed = 0
		local batchCount = 0

		-- Shuffle training data each epoch
		for i = #trainingData, 2, -1 do
			local j = math.random(i)
			trainingData[i], trainingData[j] = trainingData[j], trainingData[i]
		end

		-- Initialize gradient accumulators for batch
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

		for _, data in ipairs(trainingData) do
			local inputs = data.data
			local expectedOutputs = {}

			-- Convert label to one-hot encoding
			for i = 1, self.layerSizes[self.numLayers] do
				expectedOutputs[i] = 0
			end
			expectedOutputs[data.label] = 1

			-- Forward pass
			local outputs = self:calculateOutputs(inputs)

			-- Calculate output layer errors
			local outputErrors = {}
			for i = 1, #outputs do
				local error = expectedOutputs[i] - outputs[i]
				outputErrors[i] = error
				totalError = totalError + error * error
			end

			-- Backpropagate errors through hidden layers
			local layerErrors = {outputErrors}
			for i = self.numLayers - 1, 2, -1 do
				local layer = self.layers[i]
				local prevErrors = {}

				for j = 1, #layer.weights do
					local error = 0
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

				for j = 1, #layer.weights do
					for k = 1, #layer.weights[j] do
						local delta = errors[k] * self.activation.derivative(currentOutputs[k])
						weightUpdates[i][j][k] = weightUpdates[i][j][k] + delta * prevOutputs[j]
					end
				end

				for j = 1, #layer.biases do
					local delta = errors[j] * self.activation.derivative(currentOutputs[j])
					biasUpdates[i][j] = biasUpdates[i][j] + delta
				end
			end

			samplesProcessed += 1
			batchCount += 1

			-- Apply accumulated gradients after batch is complete
			if batchCount == batchSize then
				for i = 1, self.numLayers - 1 do
					local layer = self.layers[i]

					for j = 1, #layer.weights do
						for k = 1, #layer.weights[j] do
							layer.weights[j][k] = layer.weights[j][k] + (learningRate / batchSize) * weightUpdates[i][j][k]
							weightUpdates[i][j][k] = 0
						end
					end

					for j = 1, #layer.biases do
						layer.biases[j] = layer.biases[j] + (learningRate / batchSize) * biasUpdates[i][j]
						biasUpdates[i][j] = 0
					end
				end
				batchCount = 0
			end

			-- Yield to prevent script timeout
			if samplesProcessed % yieldEvery == 0 then
				task.wait()
			end
		end

		print("Epoch " .. epoch .. "/" .. epochs .. " - Error: " .. string.format("%.4f", totalError))
		task.wait()
	end

	print("Training complete")
end

-- Classify input and return predicted label with confidence
function NeuralNetworkService:classify(inputs)
	local outputs = self:calculateOutputs(inputs)
	local maxOutput, predictedLabel = -math.huge, -1

	for i = 1, #outputs do
		if outputs[i] > maxOutput then
			maxOutput = outputs[i]
			predictedLabel = i
		end
	end

	return predictedLabel, maxOutput, outputs
end

-- Return raw output probabilities
function NeuralNetworkService:predictProbs(inputs)
	return self:calculateOutputs(inputs)
end

-- Evaluate accuracy on test dataset
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

-- Generate confusion matrix for test dataset
function NeuralNetworkService:confusionMatrix(testData)
	local size = self.layerSizes[self.numLayers]
	local matrix = {}
	for i = 1, size do
		matrix[i] = {}
		for j = 1, size do
			matrix[i][j] = 0
		end
	end

	for _, sample in ipairs(testData) do
		local predicted = self:classify(sample.data)
		matrix[sample.label][predicted] += 1
	end

	return matrix
end

return NeuralNetworkService
