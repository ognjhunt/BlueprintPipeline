output "hunyuan_trigger_id" {
  description = "ID of the hunyuan-pipeline Eventarc trigger"
  value       = google_eventarc_trigger.hunyuan_trigger.id
}

output "sam3d_trigger_id" {
  description = "ID of the sam3d-pipeline Eventarc trigger"
  value       = google_eventarc_trigger.sam3d_trigger.id
}

output "usd_assembly_trigger_id" {
  description = "ID of the usd-assembly-pipeline Eventarc trigger"
  value       = google_eventarc_trigger.usd_assembly_trigger.id
}

output "interactive_trigger_id" {
  description = "ID of the interactive-pipeline Eventarc trigger"
  value       = google_eventarc_trigger.interactive_trigger.id
}

output "trigger_summary" {
  description = "Summary of all Eventarc triggers watching scene_assets.json"
  value = {
    hunyuan       = google_eventarc_trigger.hunyuan_trigger.name
    sam3d         = google_eventarc_trigger.sam3d_trigger.name
    usd_assembly  = google_eventarc_trigger.usd_assembly_trigger.name
    interactive   = google_eventarc_trigger.interactive_trigger.name
  }
}
