
import { useEffect, useState } from "react";
import { cn } from "@/lib/utils";
import { UserRound, Copy, ThumbsUp, ThumbsDown } from "lucide-react";
import { Button } from "@/components/ui/button";
import ReactMarkdown from 'react-markdown';

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
  const [showActions, setShowActions] = useState(false);
  const isUser = message.role === "user";

  // Process content to handle line breaks
  const processContent = (content: string) => {
    return content.replace(/\\n\\n/g, '\n\n').replace(/\\n/g, '\n');
  };

  const processedContent = processContent(message.content);

  // Typing animation for assistant messages
  useEffect(() => {
    if (isUser) {
      setDisplayedContent(processedContent);
      return;
    }

    if (currentIndex < processedContent.length) {
      const timeoutId = setTimeout(() => {
        setDisplayedContent(processedContent.slice(0, currentIndex + 1));
        setCurrentIndex(currentIndex + 1);
      }, 15);

      return () => clearTimeout(timeoutId);
    }
  }, [currentIndex, isUser, processedContent]);

  useEffect(() => {
    if (isUser) {
      setDisplayedContent(processedContent);
    } else {
      // Reset for new assistant messages
      setCurrentIndex(0);
      setDisplayedContent("");
    }
  }, [message, isUser, processedContent]);

  const handleCopy = () => {
    navigator.clipboard.writeText(processedContent);
  };

  return (
    <div
      className={cn(
        "group relative",
        isUser ? "ml-auto max-w-[80%]" : "mr-auto max-w-full"
      )}
      onMouseEnter={() => setShowActions(true)}
      onMouseLeave={() => setShowActions(false)}
    >
      <div
        className={cn(
          "flex gap-4",
          isUser ? "justify-end" : "justify-start"
        )}
      >
        {!isUser && (
          <div className="w-8 h-8 rounded-full bg-gradient-to-br from-purple-500 to-blue-500 flex items-center justify-center flex-shrink-0">
            <span className="text-sm font-medium text-white">AI</span>
          </div>
        )}
        
        <div className="max-w-full break-words">
          {isUser ? (
            <p className="whitespace-pre-wrap leading-relaxed text-white">{displayedContent}</p>
          ) : (
            <div className="markdown-content text-white">
              <ReactMarkdown
                components={{
                  p: ({ children }) => <p className="mb-3 last:mb-0 leading-relaxed text-white">{children}</p>,
                  h1: ({ children }) => <h1 className="text-xl font-bold mb-3 text-white">{children}</h1>,
                  h2: ({ children }) => <h2 className="text-lg font-bold mb-2 text-white">{children}</h2>,
                  h3: ({ children }) => <h3 className="text-base font-bold mb-2 text-white">{children}</h3>,
                  strong: ({ children }) => <strong className="font-bold text-white">{children}</strong>,
                  em: ({ children }) => <em className="italic text-white">{children}</em>,
                  ol: ({ children }) => <ol className="list-decimal list-inside mb-3 space-y-1 ml-4 text-white">{children}</ol>,
                  ul: ({ children }) => <ul className="list-disc list-inside mb-3 space-y-1 ml-4 text-white">{children}</ul>,
                  li: ({ children }) => <li className="leading-relaxed text-white">{children}</li>,
                  code: ({ children }) => <code className="bg-gray-700 px-1 py-0.5 rounded text-sm font-mono text-white">{children}</code>,
                  pre: ({ children }) => <pre className="bg-gray-700 p-3 rounded-lg overflow-x-auto mb-3 text-white">{children}</pre>,
                  blockquote: ({ children }) => <blockquote className="border-l-4 border-gray-600 pl-4 italic mb-3 text-white">{children}</blockquote>,
                }}
              >
                {displayedContent}
              </ReactMarkdown>
            </div>
          )}
          {!isUser && currentIndex < processedContent.length && (
            <span className="ml-1 inline-block w-1 h-4 bg-blue-400 animate-pulse"/>
          )}
        </div>
        
        {isUser && (
          <div className="w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center flex-shrink-0">
            <UserRound className="h-4 w-4 text-white" />
          </div>
        )}
      </div>

      {/* Action buttons for assistant messages */}
      {!isUser && showActions && currentIndex >= processedContent.length && (
        <div className="flex items-center gap-1 mt-2 ml-12 opacity-0 group-hover:opacity-100 transition-opacity">
          <Button
            variant="ghost"
            size="sm"
            className="h-8 px-2 text-gray-400 hover:text-gray-200"
            onClick={handleCopy}
          >
            <Copy className="h-3 w-3" />
          </Button>
          <Button
            variant="ghost"
            size="sm"
            className="h-8 px-2 text-gray-400 hover:text-green-400"
          >
            <ThumbsUp className="h-3 w-3" />
          </Button>
          <Button
            variant="ghost"
            size="sm"
            className="h-8 px-2 text-gray-400 hover:text-red-400"
          >
            <ThumbsDown className="h-3 w-3" />
          </Button>
        </div>
      )}
    </div>
  );
};

export default ChatMessage;
