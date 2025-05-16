
import { useState, useRef, useEffect } from "react";
import ChatMessage from "../components/ChatMessage";
import ChatInput from "../components/ChatInput";
import { toast } from "sonner";

interface Message {
  id: string;
  content: string;
  role: "user" | "assistant";
  timestamp: Date;
}

const Index = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({
      behavior: "smooth"
    });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async (content: string) => {
    if (!content.trim()) return;

    // Add user message to chat
    const userMessage: Message = {
      id: `user-${Date.now()}`,
      content,
      role: "user",
      timestamp: new Date()
    };
    setMessages(prev => [...prev, userMessage]);
    setLoading(true);
    
    try {
      const response = await fetch("https://pmogrupooscar.app.n8n.cloud/webhook-test/chat-process-pd1245", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          message: content
        })
      });
      
      // Get the response text first
      const responseText = await response.text();
      
      let assistantContent = "";
      
      // Try to parse as JSON, if it fails, use the raw text
      try {
        if (responseText && responseText.trim()) {
          const data = JSON.parse(responseText);
          assistantContent = data.output || responseText;
        } else {
          assistantContent = "Empty response from server.";
        }
      } catch (parseError) {
        // If JSON parsing fails, use the raw text as the response
        console.log("Response is not JSON, using as plain text:", responseText);
        assistantContent = responseText || "Unknown response format.";
      }

      // Add assistant response to chat
      const assistantMessage: Message = {
        id: `assistant-${Date.now()}`,
        content: assistantContent,
        role: "assistant",
        timestamp: new Date()
      };
      
      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error("Error sending message:", error);
      toast.error("Failed to send message. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-gray-900 text-white">
      {/* Header */}
      <header className="bg-gray-800 p-4 border-b border-gray-700">
        <h1 className="text-xl font-semibold text-center">AI Chat Process</h1>
      </header>

      {/* Chat Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-center space-y-4">
              <h2 className="font-bold bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent text-2xl">👋 Olá! Seja bem-vindo ao Assistente de Processos.</h2>
              <p className="text-gray-400 text-lg">Vamos organizar seus processos de forma simples e estruturada, passo a passo! </p>
            </div>
          </div>
        ) : (
          messages.map(message => <ChatMessage key={message.id} message={message} />)
        )}
        {loading && (
          <div className="flex items-center space-x-2">
            <div className="h-8 w-8 rounded-full bg-gray-700 flex items-center justify-center">
              <span className="text-sm">AI</span>
            </div>
            <div className="typing-indicator">
              <span></span>
              <span></span>
              <span></span>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Chat Input */}
      <div className="p-4 border-t border-gray-700">
        <ChatInput onSendMessage={handleSendMessage} isLoading={loading} />
      </div>
    </div>
  );
};

export default Index;
