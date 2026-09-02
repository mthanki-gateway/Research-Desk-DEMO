import { redirect } from "next/navigation";

/** Chat is the primary surface, so / goes straight there. */
export default function Home() {
  redirect("/chat");
}
