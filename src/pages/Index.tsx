
import { useState, useRef, useEffect } from "react";
import ChatMessage from "../components/ChatMessage";
import ChatInput from "../components/ChatInput";
import ProcessGuide from "../components/ProcessGuide";
import { toast } from "sonner";
import { Trash2, MoreVertical } from "lucide-react";
import { 
  DropdownMenu, 
  DropdownMenuContent, 
  DropdownMenuItem, 
  DropdownMenuTrigger 
} from "@/components/ui/dropdown-menu";

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
      const response = await fetch("https://pmogrupooscar.app.n8n.cloud/webhook/chat-process-pd1245", {
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

  const handleClearChat = () => {
    setMessages([]);
    toast.success("Conversa limpa com sucesso!");
  };

  return (
    <div className="flex flex-col md:flex-row h-screen bg-[#0f1218] text-white">
      {/* Left Side - Process Guide */}
      <div className="md:w-2/5 p-4 overflow-y-auto border-r border-gray-800">
        <div className="bg-[#131a27] p-3 rounded-md mb-4">
          <h2 className="text-lg font-medium mb-2">O que são processos? Como descrevê-los?</h2>
        </div>
        <div className="space-y-4 text-sm">
          <ProcessGuide />
        </div>
      </div>

      {/* Right Side - Chat Interface */}
      <div className="flex flex-col md:w-3/5 h-full">
        {/* Header */}
        <header className="bg-[#131a27] p-4 border-b border-gray-800 flex items-center justify-between">
          <div className="w-10">
            {/* Empty space to balance the layout */}
          </div>
          <h1 className="text-xl font-semibold text-center flex-grow">IA Chat Process</h1>
          <div className="w-10 flex justify-end">
            <DropdownMenu>
              <DropdownMenuTrigger className="p-1 rounded-md hover:bg-gray-700 focus:outline-none">
                <MoreVertical className="w-5 h-5" />
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="bg-[#1a1f2c] border-gray-700 text-white">
                <DropdownMenuItem 
                  onClick={handleClearChat}
                  className="flex items-center gap-2 cursor-pointer hover:bg-gray-700">
                  <Trash2 className="w-4 h-4" />
                  <span>Limpar conversa</span>
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </header>

        {/* Chat Messages */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {messages.length === 0 ? (
            <div className="flex items-center justify-center h-full">
              <div className="text-center max-w-xl mx-auto space-y-4">
                <h2 className="font-bold bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent text-2xl">👋 Olá! Seja bem-vindo ao Assistente de Processos.</h2>
                <p className="text-gray-400 text-lg">Vamos organizar seus processos de forma simples e estruturada, passo a passo!</p>
              </div>
            </div>
          ) : (
            messages.map(message => <ChatMessage key={message.id} message={message} />)
          )}
          {loading && (
            <div className="flex items-center space-x-2">
              <div className="h-8 w-8 rounded-full bg-gradient-to-br from-purple-500 to-blue-500 flex items-center justify-center">
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
        <div className="p-4 border-t border-gray-800">
          <ChatInput onSendMessage={handleSendMessage} isLoading={loading} />
        </div>
      </div>
    </div>
  );
};

export default Index;
