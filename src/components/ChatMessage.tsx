
import { useEffect, useState } from "react";
import { cn } from "@/lib/utils";
import { UserRound } from "lucide-react";

interface Message {
  id: string;
  content: string;
  role: "user" | "assistant";
  timestamp: Date;
}

interface ChatMessageProps {
  message: Message;
}

const ChatMessage = ({ message }: ChatMessageProps) => {
  const [displayedContent, setDisplayedContent] = useState("");
  const [currentIndex, setCurrentIndex] = useState(0);
  const isUser = message.role === "user";

  // Typing animation for assistant messages
  useEffect(() => {
    if (isUser) {
      setDisplayedContent(message.content);
      return;
    }

    if (currentIndex < message.content.length) {
      const timeoutId = setTimeout(() => {
        setDisplayedContent(message.content.slice(0, currentIndex + 1));
        setCurrentIndex(currentIndex + 1);
      }, 15);

      return () => clearTimeout(timeoutId);
    }
  }, [currentIndex, isUser, message.content]);

  useEffect(() => {
    if (isUser) {
      setDisplayedContent(message.content);
    } else {
      // Reset for new assistant messages
      setCurrentIndex(0);
      setDisplayedContent("");
    }
  }, [message, isUser]);

  return (
    <div
      className={cn(
        "flex items-start gap-3 group",
        isUser ? "justify-end" : "justify-start"
      )}
    >
      {!isUser && (
        <div className="h-8 w-8 rounded-full bg-gradient-to-br from-purple-500 to-blue-500 flex items-center justify-center flex-shrink-0 shadow-md">
          <span className="text-sm font-medium text-white">AI</span>
        </div>
      )}
      
      <div
        className={cn(
          "rounded-2xl px-4 py-2 max-w-[80%] break-words",
          isUser
            ? "bg-blue-600 text-white rounded-tr-none"
            : "bg-gray-800 text-white rounded-tl-none"
        )}
      >
        <p className="whitespace-pre-wrap">{displayedContent}</p>
        {!isUser && currentIndex < message.content.length && (
          <span className="ml-1 inline-block w-1 h-4 bg-blue-400 animate-pulse"/>
        )}
      </div>
      
      {isUser && (
        <div className="h-8 w-8 rounded-full bg-blue-600 flex items-center justify-center flex-shrink-0 shadow-md">
          <UserRound className="h-5 w-5 text-white" />
        </div>
      )}
    </div>
  );
};

export default ChatMessage;
