local DataStoreService = game:GetService("DataStoreService")
local trainingDataStore = DataStoreService:GetDataStore("TrainingData")
local events = game:GetService("ReplicatedStorage").Events
local trainingDataEvent = events.TrainingData

-- In-memory cache for training data
local cachedData = {}

local function addData(player, data, store)
	-- Initialize cache for store if needed
	if not cachedData[store] then
		cachedData[store] = {}
	end

	-- Add to cache
	table.insert(cachedData[store], data)
	print("Added sample from " .. player.Name .. ". Total: " .. #cachedData[store])

	-- Save to DataStore asynchronously
	task.spawn(function()
		pcall(function()
			local existingData = trainingDataStore:GetAsync(store) or {}
			table.insert(existingData, data)
			trainingDataStore:SetAsync(store, existingData)
		end)
	end)

	-- Send updated data to all clients
	trainingDataEvent:FireAllClients(cachedData[store])
end

-- Load existing data from DataStore on startup
task.spawn(function()
	local success, existingData = pcall(function()
		return trainingDataStore:GetAsync("Digit Recognition")
	end)

	if success and existingData then
		cachedData["Digit Recognition"] = existingData
		print("Loaded " .. #existingData .. " existing samples")
	else
		cachedData["Digit Recognition"] = {}
	end
end)

-- Send existing data when player joins
game.Players.PlayerAdded:Connect(function(player)
	player.CharacterAdded:Connect(function()
		wait(3)
		local store = "Digit Recognition"
		if cachedData[store] and #cachedData[store] > 0 then
			trainingDataEvent:FireClient(player, cachedData[store])
		end
	end)
end)

-- Listen for new training data
trainingDataEvent.OnServerEvent:Connect(addData)
