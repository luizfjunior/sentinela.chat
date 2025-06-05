import { useState, useRef, useEffect } from "react";
import ChatMessage from "../components/ChatMessage";
import ChatInput from "../components/ChatInput";
import { toast } from "sonner";
import { Trash2, MoreVertical, MessageSquare } from "lucide-react";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";
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
      const response = await fetch("https://pmogrupooscar.app.n8n.cloud/webhook/chat-sentinela-pd1245", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          message: content
        })
      });
      const responseText = await response.text();
      let assistantContent = "";
      try {
        if (responseText && responseText.trim()) {
          const data = JSON.parse(responseText);
          assistantContent = data.output || responseText;
        } else {
          assistantContent = "Empty response from server.";
        }
      } catch (parseError) {
        console.log("Response is not JSON, using as plain text:", responseText);
        assistantContent = responseText || "Unknown response format.";
      }
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
  const handleSendAudio = async (audioBlob: Blob) => {
    console.log("Sending audio:", audioBlob);

    // Add user message indicating audio was sent
    const userMessage: Message = {
      id: `user-${Date.now()}`,
      content: "🎵 Áudio enviado",
      role: "user",
      timestamp: new Date()
    };
    setMessages(prev => [...prev, userMessage]);
    setLoading(true);
    try {
      // Convert blob to base64
      const reader = new FileReader();
      reader.onload = async () => {
        const base64Audio = reader.result as string;
        const response = await fetch("https://pmogrupooscar.app.n8n.cloud/webhook/chat-sentinela-pd1245", {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify({
            audioData: base64Audio,
            messageType: "audio"
          })
        });
        const responseText = await response.text();
        let assistantContent = "";
        try {
          if (responseText && responseText.trim()) {
            const data = JSON.parse(responseText);
            assistantContent = data.output || responseText;
          } else {
            assistantContent = "Áudio processado com sucesso.";
          }
        } catch (parseError) {
          console.log("Response is not JSON, using as plain text:", responseText);
          assistantContent = responseText || "Áudio recebido e processado.";
        }
        const assistantMessage: Message = {
          id: `assistant-${Date.now()}`,
          content: assistantContent,
          role: "assistant",
          timestamp: new Date()
        };
        setMessages(prev => [...prev, assistantMessage]);
      };
      reader.readAsDataURL(audioBlob);
    } catch (error) {
      console.error("Error sending audio:", error);
      toast.error("Falha ao enviar áudio. Tente novamente.");
    } finally {
      setLoading(false);
    }
  };
  const handleClearChat = () => {
    setMessages([]);
    toast.success("Conversa limpa com sucesso!");
  };
  return <div className="flex flex-col h-screen bg-gray-50 dark:bg-[#0f1218] text-gray-900 dark:text-white">
      {/* Header */}
      <header className="sticky top-0 z-50 backdrop-blur-md border-b border-gray-200 dark:border-gray-800 bg-neutral-800">
        <div className="max-w-4xl mx-auto py-3 flex items-center justify-between px-0">
          <div className="flex items-center gap-3 mx-0">
            
            
            <h1 className="text-slate-50 py-0 px-0 mx-0 my-0 text-2xl font-semibold">Sentinela</h1>
          </div>
          
          <DropdownMenu>
            <DropdownMenuTrigger className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 focus:outline-none transition-colors">
              <MoreVertical className="w-5 h-5" />
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="border-gray-200 dark:border-gray-700 bg-neutral-900">
              <DropdownMenuItem onClick={handleClearChat} className="flex items-center gap-2 cursor-pointer bg-red-950">
                <Trash2 className="w-4 h-4" />
                <span>Limpar conversa</span>
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </header>

      {/* Chat Messages */}
      <div className="flex-1 overflow-y-auto bg-neutral-950">
        <div className="max-w-4xl mx-auto px-4">
          {messages.length === 0 ? <div className="flex items-center justify-center min-h-[60vh]">
              <div className="text-center max-w-xl mx-auto space-y-6 px-4 bg-transparent">
                <div className="bg-black">
                  <h2 className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text mb-3 text-slate-300">Bem-vindo ao Agente Sentinela Anti-Fraude!</h2>
                  
                </div>
                
                <div className="grid gap-3 mt-6">
                  
                  
                  
                </div>
              </div>
            </div> : <div className="py-6 space-y-6">
              {messages.map(message => <ChatMessage key={message.id} message={message} />)}
              {loading && <div className="flex items-start gap-4">
                  <div className="w-8 h-8 rounded-full bg-gradient-to-br from-purple-500 to-blue-500 flex items-center justify-center flex-shrink-0">
                    <span className="text-sm font-medium text-white">AI</span>
                  </div>
                  <div className="typing-indicator mt-1">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>}
            </div>}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Chat Input */}
      <div className="sticky bottom-0 backdrop-blur-md border-t border-gray-200 dark:border-gray-800 bg-zinc-900">
        <div className="max-w-4xl mx-auto px-4 py-4">
          <ChatInput onSendMessage={handleSendMessage} onSendAudio={handleSendAudio} isLoading={loading} />
        </div>
      </div>
    </div>;
};
export default Index;