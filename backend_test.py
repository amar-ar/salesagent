import requests
import unittest
import json
import os
from datetime import datetime

class UltimateSalesAssistantAPITest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(UltimateSalesAssistantAPITest, self).__init__(*args, **kwargs)
        # Get the backend URL from frontend .env file
        self.base_url = "https://4cb1cebd-146a-42dc-8113-ddf358aa4958.preview.emergentagent.com"
        self.api_url = f"{self.base_url}/api"
        self.conversation_id = None
        
    def test_01_health_check(self):
        """Test the health check endpoint"""
        print("\n🔍 Testing health check endpoint...")
        response = requests.get(f"{self.api_url}/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["groq_configured"])
        self.assertTrue(data["mongo_connected"])
        print("✅ Health check passed")
        
    def test_02_create_sample_data(self):
        """Test creating sample data"""
        print("\n🔍 Testing sample data creation...")
        response = requests.post(f"{self.api_url}/sample-data")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["message"], "Sample data created successfully")
        self.assertEqual(data["books"], 2)
        self.assertEqual(data["chunks"], 3)
        print("✅ Sample data creation passed")
        
    def test_03_get_books(self):
        """Test getting the list of books"""
        print("\n🔍 Testing book list retrieval...")
        response = requests.get(f"{self.api_url}/books")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("books", data)
        self.assertGreaterEqual(len(data["books"]), 2)  # At least the sample books
        
        # Verify book structure
        book = data["books"][0]
        self.assertIn("book_id", book)
        self.assertIn("title", book)
        self.assertIn("author", book)
        print("✅ Book list retrieval passed")
        
    def test_04_chat_functionality(self):
        """Test the chat functionality"""
        print("\n🔍 Testing chat functionality...")
        
        # Test with a simple sales question
        message = "What are the key metrics for measuring sales performance?"
        response = requests.post(
            f"{self.api_url}/chat",
            json={"message": message}
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Verify response structure
        self.assertIn("response", data)
        self.assertIn("sources", data)
        self.assertIn("actions", data)
        self.assertIn("kpis", data)
        self.assertIn("conversation_id", data)
        
        # Save conversation ID for future tests
        self.conversation_id = data["conversation_id"]
        
        # Verify response content
        self.assertGreater(len(data["response"]), 50)  # Should have substantial content
        print("✅ Chat functionality passed")
        
    def test_05_conversation_continuity(self):
        """Test conversation continuity with the same conversation ID"""
        if not self.conversation_id:
            self.skipTest("No conversation ID from previous test")
            
        print("\n🔍 Testing conversation continuity...")
        
        # Follow-up question
        message = "Can you elaborate on conversion rates?"
        response = requests.post(
            f"{self.api_url}/chat",
            json={"message": message, "conversation_id": self.conversation_id}
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Verify it's the same conversation
        self.assertEqual(data["conversation_id"], self.conversation_id)
        
        # Verify response content
        self.assertGreater(len(data["response"]), 50)
        print("✅ Conversation continuity passed")
        
    def test_06_get_conversation_history(self):
        """Test retrieving conversation history"""
        if not self.conversation_id:
            self.skipTest("No conversation ID from previous test")
            
        print("\n🔍 Testing conversation history retrieval...")
        
        response = requests.get(f"{self.api_url}/conversations/{self.conversation_id}")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Verify conversation structure
        self.assertEqual(data["conversation_id"], self.conversation_id)
        self.assertIn("messages", data)
        self.assertGreaterEqual(len(data["messages"]), 2)  # Should have at least our two messages
        print("✅ Conversation history retrieval passed")

def run_tests():
    # Create a test suite
    suite = unittest.TestSuite()
    
    # Add tests in order
    suite.addTest(UltimateSalesAssistantAPITest('test_01_health_check'))
    suite.addTest(UltimateSalesAssistantAPITest('test_02_create_sample_data'))
    suite.addTest(UltimateSalesAssistantAPITest('test_03_get_books'))
    suite.addTest(UltimateSalesAssistantAPITest('test_04_chat_functionality'))
    suite.addTest(UltimateSalesAssistantAPITest('test_05_conversation_continuity'))
    suite.addTest(UltimateSalesAssistantAPITest('test_06_get_conversation_history'))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return success/failure
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)