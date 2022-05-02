#include <vulkan/vulkan.h>
#define VK_USE_PLATFORM_WIN32_KHR
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#include <vulkan/vulkan_win32.h>

namespace Pixel {
	struct QueueFamilyIndices {
		std::optional<uint32_t> graphicsFamily;
		std::optional<uint32_t> presentFamily;

		bool IsComplete() {
			return graphicsFamily.has_value() && presentFamily.has_value();
		}
	};

	//------Swap Chain------
	struct SwapChainSupportDetails {
		VkSurfaceCapabilitiesKHR capabilities;
		std::vector<VkSurfaceFormatKHR> formats;
		std::vector<VkPresentModeKHR> presentModes;
	};
	//------Swap Chain------
	class Application {
	public:
		void Run();
		
		void RecordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex);

		bool framebufferResized = false;
	private:
		void DrawFrame();

		void InitWindow();

		void InitVulkan();

		void CleanUpSwapChain();
		void RecreateSwapChain();

		//------Init Vulkan Objects------
		void SetupDebugMessenger();
		void PickPhysicalDevice();
		void CreateLogicalDevice();
		void CreateSurface();
		void CreateImageViews();
		void CreateRenderPass();
		void CreateGraphicsPipeline();
		void CreateFramebuffers();
		void CreateCommandPool();
		void CreateVertexBuffer();
		void CreateCommandBuffers();
		void CreateSyncObjects();
		VkShaderModule CreateShaderModule(const std::vector<char>& code);
		//------aux------
		QueueFamilyIndices FindQueueFamilies(VkPhysicalDevice device);
		bool IsDeviceSuitable(VkPhysicalDevice device);
		bool CheckDeviceExtensionSupport(VkPhysicalDevice device);
		SwapChainSupportDetails QuerySwapChainSupport(VkPhysicalDevice device);
		void CreateSwapChain();
		//------swap chain information------
		VkSurfaceFormatKHR ChooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats);
		VkPresentModeKHR ChooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes);
		VkExtent2D ChooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities);
		uint32_t FindMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
		//------swap chain information------
		//------aux------
		//------Init Vulkan Objects------

		void MainLoop();

		void CleanUp();

		//------Init Vulkan Objects------
		void CreateInstance();
		//------Init Vulkan Objects------

		//---native window handle pointer, pass to swap chain create function---
		GLFWwindow* m_window;

		VkInstance m_Instance;

		//------Debug Messenger------
		VkDebugUtilsMessengerEXT debugMessenger;

		VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;

		VkDevice device;

		VkQueue graphicsQueue;
		VkQueue presentQueue;

		VkSurfaceKHR surface;
		//------Swap Chain Information------
		VkSwapchainKHR swapChain;
		std::vector<VkImage> swapChainImages;
		VkFormat swapChainImageFormat;
		VkExtent2D swapChainExtent;
		//------Swap Chain Information------

		std::vector<VkImageView> SwapChainImageViews;

		std::vector<VkFramebuffer> swapChainFramebuffers;

		//Render Pass
		VkRenderPass renderPass;

		//Pipline Layout
		VkPipelineLayout pipelineLayout;

		//Pipeline
		VkPipeline graphicsPipeline;

		//Command 
		VkCommandPool commandPool;
		std::vector<VkCommandBuffer> commandBuffers;

		//synchronization objects
		std::vector<VkSemaphore> imageAvailableSemaphores;
		std::vector<VkSemaphore> renderFinishedSemaphores;
		std::vector<VkFence>	inFlightFences;

		uint32_t currentFrame = 0;

		VkBuffer vertexBuffer;
		VkDeviceMemory vertexBufferMemory;
	};
}

