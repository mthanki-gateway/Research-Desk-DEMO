import ParleySurface from "../surface";

/**
 * The same surface as Speak, pointed at the interviewer prompt.
 *
 * Nothing else differs -- same socket, same audio handling, same turn
 * boundaries, same tools, same persistence. The mode picks a system prompt
 * server-side and separates the two lists of conversations.
 */
export default function InterviewPage() {
  return <ParleySurface mode="interview" />;
}
