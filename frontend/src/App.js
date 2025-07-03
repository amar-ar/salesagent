import React, { useState, useEffect, useRef } from 'react';
import './App.css';

const App = () => {
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [conversationId, setConversationId] = useState(null);
  const [books, setBooks] = useState([]);
  const [showUpload, setShowUpload] = useState(false);
  const [uploadFile, setUploadFile] = useState(null);
  const [uploadTitle, setUploadTitle] = useState('');
  const [uploadAuthor, setUploadAuthor] = useState('');
  const [currentResponse, setCurrentResponse] = useState(null);
  const messagesEndRef = useRef(null);

  const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8001';

  useEffect(() => {
    // Initialize with sample data and fetch books
    initializeApp();
    fetchBooks();
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const initializeApp = async () => {
    try {
      await fetch(`${BACKEND_URL}/api/sample-data`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });
    } catch (error) {
      console.error('Error initializing app:', error);
    }
  };

  const fetchBooks = async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/api/books`);
      if (response.ok) {
        const data = await response.json();
        setBooks(data.books || []);
      }
    } catch (error) {
      console.error('Error fetching books:', error);
    }
  };

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return;

    const userMessage = { type: 'user', content: inputMessage, timestamp: new Date() };
    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);

    try {
      const response = await fetch(`${BACKEND_URL}/api/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: inputMessage,
          conversation_id: conversationId,
        }),
      });

      if (response.ok) {
        const data = await response.json();
        setConversationId(data.conversation_id);
        
        const aiMessage = {
          type: 'ai',
          content: data.response,
          sources: data.sources || [],
          actions: data.actions || [],
          kpis: data.kpis || [],
          timestamp: new Date(),
        };
        
        setMessages(prev => [...prev, aiMessage]);
        setCurrentResponse(data);
      } else {
        throw new Error('Failed to send message');
      }
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage = {
        type: 'error',
        content: 'Sorry, I encountered an error. Please try again.',
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleUploadBook = async () => {
    if (!uploadFile || !uploadTitle || !uploadAuthor) {
      alert('Please fill in all fields and select a PDF file.');
      return;
    }

    const formData = new FormData();
    formData.append('file', uploadFile);
    formData.append('title', uploadTitle);
    formData.append('author', uploadAuthor);

    try {
      const response = await fetch(`${BACKEND_URL}/api/upload-book`, {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        alert(`Book "${data.title}" uploaded successfully!`);
        setShowUpload(false);
        setUploadFile(null);
        setUploadTitle('');
        setUploadAuthor('');
        fetchBooks();
      } else {
        throw new Error('Failed to upload book');
      }
    } catch (error) {
      console.error('Error uploading book:', error);
      alert('Error uploading book. Please try again.');
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const sampleQuestions = [
    "What are the key metrics for measuring sales performance?",
    "How can I improve my sales conversion rate?",
    "What's the best way to structure a sales team?",
    "How do I create a repeatable sales process?",
    "What are the psychological factors that influence buying decisions?"
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <h1 className="text-xl font-bold text-gray-900">
                  🚀 Ultimate AI Sales Assistant
                </h1>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <span className="text-sm text-gray-600">
                {books.length} books loaded
              </span>
              <button
                onClick={() => setShowUpload(true)}
                className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors"
              >
                Upload Book
              </button>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
          
          {/* Main Chat Interface */}
          <div className="lg:col-span-3">
            <div className="bg-white rounded-xl shadow-lg overflow-hidden">
              
              {/* Chat Header */}
              <div className="bg-gradient-to-r from-blue-600 to-indigo-600 text-white p-6">
                <h2 className="text-2xl font-bold mb-2">Sales Intelligence Assistant</h2>
                <p className="text-blue-100">
                  Get AI-powered insights from your sales knowledge base
                </p>
              </div>

              {/* Messages Container */}
              <div className="h-96 overflow-y-auto p-6 bg-gray-50">
                {messages.length === 0 ? (
                  <div className="text-center py-8">
                    <div className="text-gray-500 mb-4">
                      <svg className="w-16 h-16 mx-auto mb-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-3.582 8-8 8a8.955 8.955 0 01-4.336-1.121L3 20l1.121-5.664A8.955 8.955 0 013 12c0-4.418 3.582-8 8-8s8 3.582 8 8z" />
                      </svg>
                      <h3 className="text-lg font-semibold text-gray-700 mb-2">
                        Start Your Sales Conversation
                      </h3>
                      <p className="text-gray-600">
                        Ask me anything about sales strategies, metrics, or team management
                      </p>
                    </div>
                    
                    {/* Sample Questions */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mt-6">
                      {sampleQuestions.map((question, index) => (
                        <button
                          key={index}
                          onClick={() => setInputMessage(question)}
                          className="text-left p-3 bg-white rounded-lg shadow-sm border border-gray-200 hover:border-blue-300 hover:bg-blue-50 transition-colors"
                        >
                          <span className="text-sm text-gray-700">{question}</span>
                        </button>
                      ))}
                    </div>
                  </div>
                ) : (
                  <div className="space-y-4">
                    {messages.map((message, index) => (
                      <div
                        key={index}
                        className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
                      >
                        <div
                          className={`max-w-3xl px-4 py-3 rounded-lg ${
                            message.type === 'user'
                              ? 'bg-blue-600 text-white'
                              : message.type === 'error'
                              ? 'bg-red-100 text-red-800 border border-red-200'
                              : 'bg-white text-gray-800 border border-gray-200'
                          }`}
                        >
                          <p className="whitespace-pre-wrap">{message.content}</p>
                          {message.type === 'ai' && (
                            <div className="mt-3 pt-3 border-t border-gray-200">
                              {message.sources && message.sources.length > 0 && (
                                <div className="mb-3">
                                  <h4 className="font-semibold text-sm text-gray-700 mb-2">Sources:</h4>
                                  <div className="space-y-1">
                                    {message.sources.map((source, idx) => (
                                      <div key={idx} className="text-xs text-gray-600 bg-gray-50 p-2 rounded">
                                        <strong>{source.book_title}</strong> by {source.book_author}
                                        <p className="mt-1">{source.text_preview}</p>
                                      </div>
                                    ))}
                                  </div>
                                </div>
                              )}
                              {message.actions && message.actions.length > 0 && (
                                <div className="mb-3">
                                  <h4 className="font-semibold text-sm text-gray-700 mb-2">Action Items:</h4>
                                  <ul className="space-y-1">
                                    {message.actions.map((action, idx) => (
                                      <li key={idx} className="text-xs text-gray-600 flex items-start">
                                        <span className="text-green-600 mr-2">✓</span>
                                        {action}
                                      </li>
                                    ))}
                                  </ul>
                                </div>
                              )}
                              {message.kpis && message.kpis.length > 0 && (
                                <div>
                                  <h4 className="font-semibold text-sm text-gray-700 mb-2">Relevant KPIs:</h4>
                                  <div className="flex flex-wrap gap-2">
                                    {message.kpis.map((kpi, idx) => (
                                      <span key={idx} className="px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded-full">
                                        {kpi.name}
                                      </span>
                                    ))}
                                  </div>
                                </div>
                              )}
                            </div>
                          )}
                        </div>
                      </div>
                    ))}
                    {isLoading && (
                      <div className="flex justify-start">
                        <div className="bg-white border border-gray-200 rounded-lg px-4 py-3">
                          <div className="flex items-center space-x-2">
                            <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
                            <span className="text-gray-600">AI is thinking...</span>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                )}
                <div ref={messagesEndRef} />
              </div>

              {/* Input Area */}
              <div className="border-t border-gray-200 p-4 bg-white">
                <div className="flex space-x-3">
                  <input
                    type="text"
                    value={inputMessage}
                    onChange={(e) => setInputMessage(e.target.value)}
                    onKeyPress={handleKeyPress}
                    placeholder="Ask about sales strategies, metrics, team management..."
                    className="flex-1 border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    disabled={isLoading}
                  />
                  <button
                    onClick={handleSendMessage}
                    disabled={isLoading || !inputMessage.trim()}
                    className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  >
                    Send
                  </button>
                </div>
              </div>
            </div>
          </div>

          {/* Sidebar */}
          <div className="lg:col-span-1">
            <div className="space-y-6">
              
              {/* Knowledge Base */}
              <div className="bg-white rounded-xl shadow-lg p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Knowledge Base</h3>
                <div className="space-y-3">
                  {books.length === 0 ? (
                    <p className="text-gray-500 text-sm">No books uploaded yet</p>
                  ) : (
                    books.map((book) => (
                      <div key={book.book_id} className="border border-gray-200 rounded-lg p-3">
                        <h4 className="font-medium text-gray-900 text-sm">{book.title}</h4>
                        <p className="text-gray-600 text-xs mt-1">by {book.author}</p>
                        <p className="text-gray-500 text-xs mt-1">
                          {new Date(book.upload_date).toLocaleDateString()}
                        </p>
                      </div>
                    ))
                  )}
                </div>
              </div>

              {/* Quick Stats */}
              <div className="bg-white rounded-xl shadow-lg p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Quick Stats</h3>
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600">Books Loaded</span>
                    <span className="text-lg font-semibold text-blue-600">{books.length}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600">Conversations</span>
                    <span className="text-lg font-semibold text-green-600">{conversationId ? '1' : '0'}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600">Model Used</span>
                    <span className="text-xs text-gray-500">Scout/Maverick</span>
                  </div>
                </div>
              </div>

              {/* Key Features */}
              <div className="bg-white rounded-xl shadow-lg p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Key Features</h3>
                <div className="space-y-3">
                  <div className="flex items-start space-x-3">
                    <div className="w-2 h-2 bg-green-500 rounded-full mt-2"></div>
                    <div>
                      <h4 className="text-sm font-medium text-gray-900">Semantic Search</h4>
                      <p className="text-xs text-gray-600">Deep understanding of sales concepts</p>
                    </div>
                  </div>
                  <div className="flex items-start space-x-3">
                    <div className="w-2 h-2 bg-blue-500 rounded-full mt-2"></div>
                    <div>
                      <h4 className="text-sm font-medium text-gray-900">Action Items</h4>
                      <p className="text-xs text-gray-600">Actionable recommendations</p>
                    </div>
                  </div>
                  <div className="flex items-start space-x-3">
                    <div className="w-2 h-2 bg-purple-500 rounded-full mt-2"></div>
                    <div>
                      <h4 className="text-sm font-medium text-gray-900">KPI Tracking</h4>
                      <p className="text-xs text-gray-600">Sales metrics alignment</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Upload Modal */}
      {showUpload && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-xl p-6 w-full max-w-md mx-4">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Upload Sales Book</h3>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Book Title</label>
                <input
                  type="text"
                  value={uploadTitle}
                  onChange={(e) => setUploadTitle(e.target.value)}
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  placeholder="Enter book title"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Author</label>
                <input
                  type="text"
                  value={uploadAuthor}
                  onChange={(e) => setUploadAuthor(e.target.value)}
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  placeholder="Enter author name"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">PDF File</label>
                <input
                  type="file"
                  accept=".pdf"
                  onChange={(e) => setUploadFile(e.target.files[0])}
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
            </div>
            <div className="flex justify-end space-x-3 mt-6">
              <button
                onClick={() => setShowUpload(false)}
                className="px-4 py-2 text-gray-700 bg-gray-200 rounded-lg hover:bg-gray-300 transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleUploadBook}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
              >
                Upload
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default App;