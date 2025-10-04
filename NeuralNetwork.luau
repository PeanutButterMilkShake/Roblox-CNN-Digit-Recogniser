local ReplicatedStorage = game:GetService("ReplicatedStorage")
local UserInputService = game:GetService("UserInputService")
local NeuralNetworkService = require(ReplicatedStorage.Services.NeuralNetworkService)

local trainingData = {}
local events = ReplicatedStorage.Events
local trainingDataEvent = events.TrainingData
local grid = script.Parent.Background.Grid
local gridSize = Vector2.new(30, 30)
local mouseInput = ""
local isTraining = false

workspace.CurrentCamera.CameraType = Enum.CameraType.Scriptable

local network = NeuralNetworkService.new({900, 300, 10}, "Sigmoid")

-- convert grid to input data for network
local function getData()
	local data = {}
	for y = 0, gridSize.y - 1 do
		for x = 0, gridSize.x - 1 do
			local cell = grid:FindFirstChild(x .. "," .. y)
			if cell then
				data[#data+1] = 1 - cell.BackgroundColor3.R
			end
		end
	end
	return data
end

-- handle submit button to add training sample
script.Parent.Background.NetworkFrame.TextButton.Activated:Connect(function()
	local labelText = script.Parent.Background.NetworkFrame.TextBox.Text
	local label = tonumber(labelText)
	if label and label >= 1 and label <= 10 then
		trainingDataEvent:FireServer({label = label, data = getData()}, "Digit Recognition")

		for _, cell in grid:GetChildren() do
			if cell:IsA("Frame") then
				cell.BackgroundColor3 = Color3.new(1,1,1)
			end
		end
		script.Parent.Background.NetworkFrame.TextBox.Text = ""
	end
end)

-- predict digit from current grid
local function predictDigit()
	if isTraining then return 0,0,{} end
	return network:classify(getData())
end

-- update UI with prediction and confidence
local function givePrediction()
	if isTraining then
		script.Parent.Background.NetworkFrame.Number.Text = "Training..."
		script.Parent.Background.NetworkFrame.Confidence.Text = ""
		return
	end

	local digit, confidence, allOutputs = predictDigit()
	script.Parent.Background.NetworkFrame.Number.Text = tostring(digit)
	script.Parent.Background.NetworkFrame.Confidence.Text = "Confidence: " .. math.floor(confidence*100+0.5) .. "%"

	-- print all outputs
	print("All Output Confidences:")
	for i=1,#allOutputs do
		print("Digit "..i..": "..math.floor(allOutputs[i]*100+0.5).."%")
	end
end

-- color a single cell based on mouse input
local function colorCell(cell: Frame)
	if mouseInput == "" or isTraining then return end
	cell.BackgroundColor3 = mouseInput == "right" and Color3.new(1,1,1) or Color3.new(0,0,0)
	givePrediction()
end

-- create grid cells
local function generateGrid()
	local cellWidth, cellHeight = 1/gridSize.x, 1/gridSize.y
	for x=0,gridSize.x-1 do
		for y=0,gridSize.y-1 do
			local cell = Instance.new("Frame")
			cell.Parent = grid
			cell.Position = UDim2.new(x*cellWidth,0,y*cellHeight,0)
			cell.Size = UDim2.new(cellWidth,0,cellHeight,0)
			cell.BackgroundColor3 = Color3.new(1,1,1)
			cell.BorderSizePixel = 0
			cell.Name = x..","..y
			cell.MouseEnter:Connect(function() colorCell(cell) end)
		end
	end
end

-- handle mouse input
UserInputService.InputBegan:Connect(function(input, gameProcessed)
	if gameProcessed then return end
	if input.UserInputType == Enum.UserInputType.MouseButton1 then
		mouseInput = "left"
	elseif input.UserInputType == Enum.UserInputType.MouseButton2 then
		mouseInput = "right"
	elseif input.UserInputType == Enum.UserInputType.MouseButton3 then
		-- middle click clears grid
		for _,cell in grid:GetChildren() do
			if cell:IsA("Frame") then
				cell.BackgroundColor3 = Color3.new(1,1,1)
			end
		end
		givePrediction()
	end
end)

UserInputService.InputEnded:Connect(function(input)
	if input.UserInputType == Enum.UserInputType.MouseButton1 or input.UserInputType == Enum.UserInputType.MouseButton2 then
		mouseInput = ""
	end
end)

-- receive training data from server and start training
trainingDataEvent.OnClientEvent:Connect(function(data)
	trainingData = data
	print("Received "..#trainingData.." training samples")
	if #trainingData > 0 and not isTraining then
		isTraining = true
		givePrediction()
		task.spawn(function()
			print("Starting training...")
			network:train(trainingData, 100, 1, 5)
			local accuracy = network:evaluate(trainingData)
			print("Training complete. Accuracy on training set: " .. math.floor(accuracy*100+0.5) .. "%")
			isTraining = false
			givePrediction()
		end)
	end
end)
-- initialize grid
generateGrid()
